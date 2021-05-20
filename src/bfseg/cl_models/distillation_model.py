import tensorflow as tf
from tensorflow import keras
import warnings

from bfseg.cl_models import BaseCLModel
from bfseg.utils.models import create_model


class DistillationModel(BaseCLModel):
  r"""Model with distillation loss. Two types of distillation loss are possible:
  1. Feature distillation on the intermediate feature space (at the output of
     the encoder);
  2. Distillation on the network output.

  Args:
    run (sacred.run.Run): Object identifying the current sacred run.
    root_output_dir (str): Path to the folder that will contain the experiment
      logs and the saved models.
  """

  def __init__(self, run, root_output_dir):
    if (run.config['cl_params']['pretrained_dir'] is None):
      raise KeyError(
          "Pre-trained weights must be specified when using distillation.")
    try:
      self._lambda_distillation = run.config['cl_params']['lambda_distillation']
      self._lambda_type = run.config['cl_params']['lambda_type']
      if (self._lambda_type == "both_ce_and_regularization"):
        if (not (isinstance(self._lambda_distillation, float) and
                 0. <= self._lambda_distillation <= 1.)):
          raise ValueError(
              "The parameter `lambda_distillation` must be a float between 0.0 "
              "and 1.0.")
      elif (self._lambda_type == "regularization_only"):
        if (not (isinstance(self._lambda_distillation, float) and
                 self._lambda_distillation >= 0.)):
          raise ValueError(
              "The parameter `lambda_distillation` must be a non-negative "
              "float.")
      else:
        raise KeyError("The CL parameter `lambda_type` must be one of: "
                       "'both_ce_and_regularization', 'regularization_only'.")
    except KeyError:
      raise KeyError(
          "Distillation model requires the CL parameters `lambda_distillation` "
          "and `lambda_type` to be specified.")

    try:
      self._distillation_type = run.config['cl_params']['distillation_type']
      if (self._distillation_type not in ["feature", "output"]):
        raise ValueError(
            f"Invalid value {self._distillation_type} for parameter "
            "`distillation_type`. Valid values are: 'feature', 'output'.")
    except KeyError:
      raise KeyError(
          "Distillation model requires the CL parameter `distillation_type` "
          "to be specified.")

    super(DistillationModel, self).__init__(run=run,
                                            root_output_dir=root_output_dir)

  def _build_model(self):
    r"""Builds the models. Overrides method from base class.
    """
    # Create the trainable `new_model`, which has the two outputs `encoder` and
    # `model`.
    super()._build_model()
    # Create the fixed model from which distillation will be performed.
    if (self._distillation_type == "feature"):
      self.old_encoder, _ = create_model(
          model_name=self.run.config['network_params']['architecture'],
          freeze_encoder=True,
          # NOTE: here it would be enough to just freeze the encoder, since it
          # is the only part used in feature distillation.
          freeze_whole_model=True,
          normalization_type=self.run.config['network_params']
          ['normalization_type'],
          **self.run.config['network_params']['model_params'])
    elif (self._distillation_type == "output"):
      _, self.old_model = create_model(
          model_name=self.run.config['network_params']['architecture'],
          freeze_encoder=True,
          freeze_whole_model=True,
          normalization_type=self.run.config['network_params']
          ['normalization_type'],
          **self.run.config['network_params']['model_params'])

  def _build_loss_and_metric(self):
    r"""Adds loss criteria and metrics. Overrides the parent method.
    """
    # Create the losses and metrics from the base class.
    super()._build_loss_and_metric()
    # Add the distillation loss, simply defined as a mean-square-error loss.
    self.loss_distillation = keras.losses.MeanSquaredError()

  def forward_pass(self, training, x, y, mask):
    r"""Forward pass. Overrides the parent method.

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
      loss (dict): Losses from performing the forward pass.
    """
    y_masked = tf.boolean_mask(y, mask)
    if (self._distillation_type == "feature"):
      [pred_feature, pred_y] = self.new_model(x, training=True)
      pred_y_masked = tf.boolean_mask(pred_y, mask)
      old_feature = self.old_encoder(x, training=False)
      distillation_loss = self.loss_distillation(tf.stop_gradient(old_feature),
                                                 pred_feature)
    elif (self._distillation_type == "output"):
      [_, pred_y] = self.new_model(x, training=True)
      pred_y_masked = tf.boolean_mask(pred_y, mask)
      pseudo_y = self.old_model(x, training=False)
      pseudo_y = tf.argmax(pseudo_y, axis=-1)
      pseudo_y_masked = tf.boolean_mask(pseudo_y, mask)
      distillation_loss = self.loss_ce(tf.stop_gradient(pseudo_y_masked),
                                       pred_y_masked)
    output_loss = self.loss_ce(y_masked, pred_y_masked)
    if (self._lambda_type == "both_ce_and_regularization"):
      loss = (1 - self._lambda_distillation
             ) * output_loss + self._lambda_distillation * distillation_loss
    else:
      loss = output_loss + self._lambda_distillation * distillation_loss

    # Return also the distillation loss for tracking.
    loss = {
        'loss': loss,
        'cross_entropy_loss': output_loss,
        'distillation_loss': distillation_loss
    }

    return pred_y, pred_y_masked, y_masked, loss
