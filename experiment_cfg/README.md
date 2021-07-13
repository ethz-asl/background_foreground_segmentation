# Description of training parameters

## Network parameters

### Required parameters

- `architecture` (`str`): Architecture type. Valid values are:
  - `"fast_scnn"`: Fast-SCNN architecture.
  - `"unet"`: U-Net architecture.
- `freeze_encoder` (`bool`): Whether the encoder should be frozen or not. If `True`, the CL parameter `pretrained_dir` must be specified. Will raise an error if `True` and CL framework used (CL parameter `cl_framework`) is not in: `distillation`, `finetune`.
- `image_h` (`int`): Image height.
- `image_w` (`int`): Image width.
- `normalization_type`: (`str`): Type of normalization to use in the model, if normalization layers are present. Valid values are:
  - `"batch"` (batch normalization)
  - `"group"` (group normalization). Only supported with architecture `"fast_scnn"`.

### Optional parameters

Optional parameters used to instantiate the selected architecture (cf. `architecture`). Valid parameters can be found where the architectures are defined (cf. `src/bfseg/models/__init__.py` to check from where the model definitions are imported).

____

## Training parameters

### Required parameters

- `batch_size` (`int`): Batch size.
- `learning_rate` (`float`): Learning rate.

- `num_training_epochs` (`int`): Number of training epochs.
- `perform_data_augmentation` (`bool`): Whether or not to perform data augmentation.
- `reduce_lr_on_plateau` (`bool`): Whether or not to use learning-rate reduction on plateau.
- `stopping_min_epoch` (`int`): Minimum epoch after which early stopping can be performed.
- `stopping_patience` (`int`): Patience parameter of the early-stopping callback.
- `use_balanced_loss` (`bool`): Whether or not balanced cross-entropy loss should be used (cf. `src/bfseg/utils/metrics.py`).


### Optional parameters

None.

### Required parameters for multi-task setup 

We add four additional `depth_params` multi-task setup:
- `semseg_weight` (`float`): Weight of cross-entropy loss for semantic segmentation
- `depth_weigth` (`float`): Weight of depth loss for depth regression
- `consistency_weight` (`float`): Weight of consistency loss betweeen segementation and depth
- `preprocessing_mode` (`str`): Preprocessing mode of depth labels. Available options: `power_std`, `power_median`, `inverse_std`, `inverse_median`, `inverse`, `normal` (normal for DORN)

____

## Dataset parameters

### Required parameters
- `replay_datasets` (`list` of `str`, or `None`): If not `None`, name of the replay dataset(s) to use.
- `replay_datasets_scene` (`list` of `str`, or `None`): If not `None`, scene type of each of the replay dataset(s) to use.
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
- `BfsegValidationLabeled`. Valid dataset scenes:
  - `None`
  - `"ARCHE"`
  - `"CLA"`
- `MeshdistPseudolabels`. Valid dataset scenes:
  - `None`
  - `"garage_full"`
  - `"rumlang_full"`
  - `"garage1"`
  - `"garage2"`
  - `"garage3"`
  - `"office4"`
  - `"office4_2302"`
  - `"office4_2402"`
  - `"office5"`
  - `"office6_2502"`
  - `"rumlang2"`
  - `"rumlang3"`
- `MeshdistPseudolabelsDyn`. Valid dataset scenes:
  - `"rumlang1_full_dyn"`
  - `"rumlang1_full"`
- `MeshdistPseudolabelsDense`. Valid dataset scenes: 
  - `"office12_dense20_dyn_cam2"` 
  - `"office12_sparse50_dyn_cam2"`
  - `"office12_sparse50_dyn_allcams"`
  - `"office3_dense20_dyn_cam2"` (used for dense pseudo-labels in semester project results)
  - `"office3_sparse50_dyn_cam2"` (used for sparse pseudo-labels in semester project results)
  - `"office3_sparse50_dyn_allcams"`
  - `"office3_combined2050_dyn_cam2"` (used for combined pseudo-label in semester project results)
- `MeshdistPseudolabelsDenseDepth`. Valid dataset scenes: 
  - `"office12_densedepth20_dyn_cam2"` 
  - `"office3_densedepth20_dyn_cam2"` (used for mulit-task dense pseudo-labels in semester project results)
  - `None`
  - `"OFFICE"`
  - `"RUMLANG"`
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
  - `"distillation"`: Distillation model. Requires both the `pretrained_dir` argument to be not `None`, and the `lambda_distillation` argument to be specified.
  - `"ewc"`: EWC. Requires both the `pretrained_dir` argument to be not `None`, and the `lambda_ewc` argument to be specified.
  - `"finetune"`: Fine-tuning, using the pre-trained model weights in `pretrained_dir`. If no `pretrained_dir` is specified, training is performed from scratch.
- `fraction_replay_ds_to_use` (`float` or `None`): Cf. `src/bfseg/utils/replay_buffer.py`.
- `pretrained_dir` (`str`): Directory containing the pre-trained model weights. If `None`, no weights are loaded.
- `ratio_main_ds_replay_ds` (`list` of `int` or `None`): Cf. `src/bfseg/utils/replay_buffer.py`.

### Optional parameters

- `distillation_type` (`str`): Required if using `"distillation"` as `cl_framework`. Valid values are:
  - `"feature"`: Feature on the intermediate feature space (at the output of the encoder);
  - `"output"`: Distillation on the network output.
- `ewc_fisher_params_use_gt` (`bool`): Required if using `"ewc"` as `cl_framework`. If `True`, the Fisher matrix uses the ground-truth labels to compute the log-likelihoods; if `False`, it uses the class to which the network assigns the most likelihood.
- `lambda_distillation` (`float`): Required if using `"distillation"` as `cl_framework`. Regularization hyperparameter used to weight the loss.  Valid values are between 0 and 1. In particular, the loss is computed as:
  - `(1 - lambda_distillation) * loss_ce + lambda_distillation * consolidation_loss`, if `lambda_type` is `"both_ce_and_regularization"`;
  - `loss_ce + lambda_distillation * consolidation_loss`, if `lambda_type` is `"regularization_only"`.

  In both cases:
  - `loss_ce` is the cross-entropy loss computed on the current task;
  - `distillation_loss` is the distillation loss computed between the feature/output of the current network and of the teacher network.
- `lambda_ewc` (`float`): Required if using `"ewc"` as `cl_framework`. Regularization hyperparameter used to weight the loss.  Valid values are between 0 and 1. In particular, the loss is computed as:
  - `(1 - lambda_ewc) * loss_ce + lambda_ewc * consolidation_loss`, if `lambda_type` is `"both_ce_and_regularization"`;
  - `loss_ce + lambda_ewc * consolidation_loss`, if `lambda_type` is `"regularization_only"`;

  In both cases:
  - `loss_ce` is the cross-entropy loss computed on the current task;
  - `consolidation_loss` is the regularization loss on the parameters from the previous task.
- `lambda_type` (`str`): Required if using `"distillation"` or `"ewc"` as `cl_framework`. Valid values are: `"both_ce_and_regularization"`, `"regularization_only"`. Cf. `lambda_distillation` and `lambda_ewc`.