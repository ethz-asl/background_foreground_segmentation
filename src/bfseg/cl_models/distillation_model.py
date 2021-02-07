import tensorflow as tf
from tensorflow import keras
import warnings

from bfseg.cl_models import BaseCLModel


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
      if (not (isinstance(self._lambda_distillation, float) and
               0. <= self._lambda_distillation <= 1.)):
        raise ValueError(
            "The parameter `lambda_ewc` must be a float between 0.0 and 1.0.")
    except KeyError:
      raise KeyError(
          "Distillation model requires the CL parameter `lambda_distillation` "
          "to be specified.")

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

    self._started_training_new_task = False

  def _build_model(self):
    r"""Builds the models. Overrides method from base class.
    """
    super()._build_model()
    assert (self.run.config['cl_params']['cl_framework']
            in ["ewc", "finetune"
               ]), "Currently, only EWC and fine-tuning are supported."
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
      distillation_loss = self.loss_mse(tf.stop_gradient(old_feature),
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
    loss = (1 - self.config.lambda_distillation
           ) * output_loss + self.config.lambda_distillation * distillation_loss

    # Return also the distillation loss for tracking.
    loss = {
        'loss': loss,
        'cross_entropy_loss': output_loss,
        'distillation_loss': distillation_loss
    }

    return pred_y, pred_y_masked, y_masked, loss
