# Description of training parameters

## Network parameters

### Required parameters

- `architecture` (`str`): Architecture type. Valid values are:
  - `'fast_scnn`': Fast-SCNN architecture.
  - `'unet'`: U-Net architecture.
- `image_h` (`int`): Image height.
- `image_w` (`int`): Image width.

### Optional parameters

Optional parameters used to instantiate the selected architecture (cf. `architecture`). Valid parameters can be found where the architectures are defined (cf. `src/bfseg/models/__init__.py` to check from where the model definitions are imported).

____

## Training parameters

### Required parameters

- `batch_size` (`int`): Batch size.
- `learning_rate` (`float`): Learning rate.

- `num_training_epochs` (`int`): Number of training epochs.
- `stopping_patience` (`int`): Patience parameter of the early-stopping callback.

### Optional parameters

None.

____

## Dataset parameters

### Required parameters

- `test_dataset` (`str`): Name of the test dataset.

- `test_scene` (`str`): Scene type of the test dataset.
- `train_dataset` (`str`): Name of the training dataset.

- `train_scene` (`str`): Scene type of the training dataset.

- `validation_percentage` (`int`): Percentage of the training scene to use for validation.

### Optional parameters

- `fisher_params_dataset` (`str`): Name of the dataset to be used to compute the Fisher information matrix in EWC. Required if the CL parameter `cl_framework` is `ewc`.
- `fisher_params_scene` (`str`): Scene type of the dataset to be used to compute the Fisher information matrix in EWC. Required if the CL parameter `cl_framework` is `ewc`.
- `fisher_params_sample_percentage` (`int`): Percentage of samples to be randomly selected from the dataset `fisher_params_dataset` and used to compute the Fisher information matrix in EWC. Required if the CL parameter `cl_framework` is `ewc`.

### Valid dataset names

- `BfsegCLAMeshdistLabels`. Valid dataset scenes:
  - `None`
  - `"kitchen"`
  - `"bedroom"`
- `NyuDepthV2Labeled`. Valid dataset scenes:
  - `None`

____

## Logging parameters

### Required parameters

- `exp_name` (`str`): Name of the current experiment.
- `model_save_freq` (`int`): Frequency (in epochs) for saving models.

### Optional parameters

None.

____

## CL parameters

### Required parameters

- `cl_framework` (`str`): CL framework to use. Valid values are:
  - `"ewc"`: EWC. Requires both the `pretrained_dir` argument to be not `None`, and the `lambda_ewc` argument to be specified.
  - `"finetune"`: Fine-tuning, using the pre-trained model weights in `pretrained_dir`. If no `pretrained_dir` is specified, training is performed from scratch.
- `pretrained_dir` (`str`): Directory containing the pre-trained model weights. If `None`, no weights are loaded.

### Optional parameters

- `ewc_fisher_params_use_gt` (`bool`): Required if using `"ewc"` as `cl_framework`. If `True`, the Fisher matrix uses the ground-truth labels to compute the log-likelihoods; if `False`, it uses the class to which the network assigns the most likelihood.
- `lambda_ewc` (`float`): Required if using `"ewc"` as `cl_framework`. Regularization hyperparameter used to weight the loss.  Valid values are between 0 and 1. In particular, the loss is computed as: `(1 - lambda_ewc) * loss_ce + lambda_ewc * consolidation_loss`, where
  - `loss_ce` is the cross-entropy loss computed on the current task;
  - `consolidation_loss` is the regularization loss on the parameters from the previous task.