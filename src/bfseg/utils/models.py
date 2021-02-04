""" Utils to instantiate models compatible with a generic CL experiment.
"""
import inspect
import warnings

from bfseg.models import FastSCNN, UNet


def create_model(model_name, pretrained_dir, image_h, image_w, **model_params):
  r"""Factory function that creates a model with the given parameters.

  Args:
    model_name (str): Type of architecture to use. Valid values are: "unet".
    pretrained_dir (str): If not None, path to the pretrained weights to load.
    image_h (int): Image height.
    image_w (int): Image width.
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
    model_fn = UNet
  else:
    raise ValueError(
        f"Invalid model name {model_name}. Valid values are: 'unet'.")
  # For all models, fix the input shape and the number of classes.
  model_params['classes'] = 2
  model_params['input_shape'] = (image_h, image_w, 3)

  if (hasattr(model_fn, '__wrapped__')):
    # Models from `segmentation_models` are instantiated through wrapping
    # functions.
    valid_params = inspect.getfullargspec(model_fn.__wrapped__).args
    default_param_values = inspect.getfullargspec(model_fn.__wrapped__).defaults
  else:
    valid_params = inspect.getfullargspec(model_fn).args
    default_param_values = inspect.getfullargspec(model_fn).defaults

  # For each valid parameter, pair it with its default value (None, if the
  # parameter is non-optional).
  num_nonoptional_params = len(valid_params) - len(default_param_values)
  assert (num_nonoptional_params >= 0)
  nonoptional_params = valid_params[:num_nonoptional_params]
  optional_params = valid_params[num_nonoptional_params:]
  default_param_values = [None for _ in range(num_nonoptional_params)
                         ] + list(default_param_values)
  valid_params_with_defaults = dict(zip(valid_params, default_param_values))

  # Check that all non-optional parameters are present.
  missing_nonoptional_params = set(nonoptional_params) - set(model_params)
  if (len(missing_nonoptional_params) > 0):
    raise TypeError(
        f"Model '{model_name}' missing the following required positional "
        f"arguments: {missing_nonoptional_params}.")

  #TODO(fmilano): Log default values used.

  # Ignore invalid input parameters.
  invalid_input_parameters = set(model_params) - set(valid_params)
  if (len(invalid_input_parameters) > 0):
    warnings.warn(
        f"Ignoring the following parameters, invalid for model '{model_name}': "
        f"{invalid_input_parameters}.")
    for invalid_param in invalid_input_parameters:
      model_params.pop(invalid_param)

  # Merge model parameters and default parameters.
  model_params = {
      key: model_params.get(key, valid_params_with_defaults[key])
      for key in valid_params
  }

  encoder, model = model_fn(**model_params)

  return encoder, model
