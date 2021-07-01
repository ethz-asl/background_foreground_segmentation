# Dense Experiments Commands

## Prepare
SSH into Xavier: 

```bash
ssh xavier@<IPAdress>
```

Source environment:

```bash
source ~/envs/bfs1/bin/activate
```

Update background foreground segmentation directory:
```bash
cd ~/git/background_foreground_segmentation
git pull
```
Pre-trained models are stored on Xavier under /xavier_ssd/saved_weights/ and available under https://drive.google.com/drive/folders/1fQSzlmditIgdCHuZXNRsEpVQu0-whQSP?usp=sharing)

## Additional Parameters:
Generally, the same parameters are used as explained in README.md
We add four additional `depth_params` multi-task setup:
- `semseg_weight` (`float`): Weight of cross-entropy loss for semantic segmentation
- `depth_weigth` (`float`): Weight of depth loss for depth regression
- `consistency_weight` (`float`): Weight of consistency loss betweeen segementation and depth
- `preprocessing_mode` (`str`): Preprocessing mode of depth labels. Available options: `power_std`, `power_median`, `inverse_std`, `inverse_median`, `inverse`, `normal` (for DORN)


## Single Task 
Pretraining on NYU:
```bash
python src/train_binary_segmodel_base.py with experiment_cfg/pretrain_dense_labels/fast_scnn_nyu_finetune.yml
```
Finetuning with Sparse Labels (Office):
```bash
python src/train_binary_segmodel_base.py with experiment_cfg/finetune_dense_labels/fast_scnn_from_nyu_to_office3_sparse50_dyn_cam2_finetune1.yml
```
Finetuning with Dense Labels (Office):
```bash
python src/train_binary_segmodel_base.py with experiment_cfg/finetune_dense_labels/fast_scnn_from_nyu_to_office3_dense20_dyn_cam2_finetune1.yml
```
Finetuning with Combined Labels (Office):
```bash
python src/train_binary_segmodel_base.py with experiment_cfg/finetune_dense_labels/fast_scnn_from_nyu_to_office3_combined2050_dyn_cam2_finetune1.yml
```

## Multi-Task
Pretraining on NYU:
```bash
# Power + Std
python src/train_binary_segmodel_depth.py with experiment_cfg/pretrain_dense_labels/fast_scnn_plus_depth_nyu_finetune_power_std.yml
# Power + Median
python src/train_binary_segmodel_depth.py with experiment_cfg/pretrain_dense_labels/fast_scnn_plus_depth_nyu_finetune_power_median.yml
# Inverse + Std
python src/train_binary_segmodel_depth.py with experiment_cfg/pretrain_dense_labels/fast_scnn_plus_depth_nyu_finetune_inverse_std.yml
# Inverse + Median
python src/train_binary_segmodel_depth.py with experiment_cfg/pretrain_dense_labels/fast_scnn_plus_depth_nyu_finetune_inverse_median.yml
# Power + Std (no consistency loss)
python src/train_binary_segmodel_depth.py with experiment_cfg/pretrain_dense_labels/fast_scnn_plus_depth_nyu_finetune_power_std_nocons.yml
# Power + Std (no depth loss)
python src/train_binary_segmodel_depth.py with experiment_cfg/pretrain_dense_labels/fast_scnn_plus_depth_nyu_finetune_power_std_nodepth.yml
# DORN
python src/train_binary_segmodel_dorn.py with experiment_cfg/pretrain_dense_labels/fast_scnn_plus_dorn_nyu_finetune.yml
```

Finetuning on Office:
```bash
# Power + Std
python src/train_binary_segmodel_depth.py with experiment_cfg/finetune_dense_labels/fast_scnn_from_nyu_to_office3_densedepth20_dyn_cam2_finetune1_power_std.yml
# Power + Median
python src/train_binary_segmodel_depth.py with experiment_cfg/finetune_dense_labels/fast_scnn_from_nyu_to_office3_densedepth20_dyn_cam2_finetune1_power_median.yml
# Inverse + Std
python src/train_binary_segmodel_depth.py with experiment_cfg/finetune_dense_labels/fast_scnn_from_nyu_to_office3_densedepth20_dyn_cam2_finetune1_inverse_std.yml
# Inverse + Median
python src/train_binary_segmodel_depth.py with experiment_cfg/finetune_dense_labels/fast_scnn_from_nyu_to_office3_densedepth20_dyn_cam2_finetune1_inverse_median.yml
# Power + Std (no consistency loss)
python src/train_binary_segmodel_depth.py with experiment_cfg/finetune_dense_labels/fast_scnn_from_nyu_to_office3_densedepth20_dyn_cam2_finetune1_power_std_nocons.yml
# Power + Std (no depth loss)
python src/train_binary_segmodel_depth.py with experiment_cfg/finetune_dense_labels/fast_scnn_from_nyu_to_office3_densedepth20_dyn_cam2_finetune1_power_std_nodepth.yml
# DORN
python src/train_binary_segmodel_dorn.py with experiment_cfg/finetune_dense_labels/fast_scnn_from_nyu_to_office3_densedepth20_dyn_cam2_finetune1_dorn.yml
```