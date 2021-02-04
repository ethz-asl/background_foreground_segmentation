""" Utils to instantiate models compatible with a generic CL experiment.
"""
import segmentation_models as sm
import warnings


def create_model(model_name, pretrained_dir, **model_params):
  r"""Factory function that creates a model with the given parameters.

  Args:
    model_name (str): Type of architecture to use. Valid values are: "unet".
    pretrained_dir (str): If not None, path to the pretrained weights to load.
    ---
    model_params: Parameters required to instantiate the model. Cf. basic config
      file for list of valid parameters depending on the architecture.

  Returns:
    model (tf.keras.Model): Output of the model instantiated with the given
      parameters.
    encoder (tf.keras.Model): Output of the network after an encoder or similar,
      if available. Used to apply intermediate distillation. If not available,
      None is returned.
  """
  if (model_name == "unet"):
    valid_params = ['backbone_name', 'image_h', 'image_w']
    for param in model_params.keys():
      if (param not in valid_params):
        warnings.warn(
            f"Ignoring parameter {param}, invalid for model {model_name}.")
    encoder, model = sm.Unet(backbone_name=model_params['backbone_name'],
                             input_shape=(model_params['image_h'],
                                          model_params['image_w'], 3),
                             classes=2,
                             activation='sigmoid',
                             weights=pretrained_dir,
                             encoder_freeze=False)
  else:
    raise ValueError(
        f"Invalid model name {model_name}. Valid values are: 'unet'.")

  return encoder, model
