def default_config():
  r"""Default configuration for fine-tuning experiment on NYU with U-Net.
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
      'learning_rate': 1e-4,
      'num_training_epochs': 40,
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
      'model_save_freq': 1,
      'exp_name': "exp_stage1"
  }

  # CL parameters.
  #TODO(fmilano): Add factory method for CL models.
  cl_params = {'cl_framework': "finetune", 'pretrained_dir': None}