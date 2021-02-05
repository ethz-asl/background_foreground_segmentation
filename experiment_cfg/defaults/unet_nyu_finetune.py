def default_config():
  r"""Default configuration for base segmentation experiments.
  - Network parameters:
    - architecture (str): Architecture type. Valid values are:
      - "unet": U-Net architecture. Required parameters are:
        - backbone_name (str): Name of the backbone of the U-Net architecture.
        - image_h (int): Image height.
        - image_w (int): Image width.
  - Training parameters:
    - batch_size (int): Batch size.
    - learning_rate (float): Learning rate.
    - num_training_epochs (int): Number of training epochs.
    - stopping_patience (int): Patience parameter of the early-stopping
        callback.
  - Dataset parameters:
    - test_dataset (str): Name of the test dataset.
    - test_scene (str): Scene type of the test dataset. Valid values are: None,
        "kitchen", "bedroom".
    - train_dataset (str): Name of the training dataset.
    - train_scene (str): Scene type of the training dataset. Valid values are:
        None, "kitchen", "bedroom".
    - validation_percentage (int): Percentage of the training scene to use for
      validation.
  - Logging parameters:
    - exp_name (str): Name of the current experiment.
    - metric_log_frequency (str): Frequency with which the training metrics are
      logged. Valid values are "epoch" (i.e., every epoch), "batch" (i.e., every
      batch).
    - model_save_freq (int): Frequency (in epochs) for saving models.
  - CL parameters:
    - cl_framework (str): CL framework to use. Valid values are:
      - "finetune": Fine-tuning, using the pretrained model weights in
        `pretrained_dir`. If no `pretrained_dir` is specified, training is
        performed from scratch.
    - pretrained_dir (str): Directory containing the pretrained model weights.
  """
  # Network parameters.
  network_params = {
      'architecture': 'unet',
      'model_params': {
          'backbone_name': "vgg16",
          #TODO (fmilano): Retrieve from first training sample.
          'image_h': 480,
          'image_w': 640
      }
  }

  # Training parameters.
  training_params = {
      'batch_size': 8,
      'learning_rate': 1e-5,
      'num_training_epochs': 3,
      'stopping_patience': 100
  }

  # Dataset parameters.
  dataset_params = {
      'test_dataset': "NyuDepthV2Labeled",
      'test_scene': None,
      'train_dataset': "BfsegCLAMeshdistLabels",
      'train_scene': None,
      'validation_percentage': 20
  }

  # Logging parameters.
  logging_params = {
      'metric_log_frequency': "batch",
      'model_save_freq': 1,
      'exp_name': "exp_stage1"
  }

  # CL parameters.
  #TODO(fmilano): Add factory method for CL models.
  cl_params = {'cl_framework': "finetune", 'pretrained_dir': None}