# Background Foreground Segementation

Structure allows for standalone python useage and useage within ROS.

To use the python package `cd src; pip install -e .`
## Pseudo Labels using distance to mesh

### Dataset Creator (ROS)
#### Overview
This package extracts the following information from a ROS Bag:

- For each camera:
    - Camera information (All information from the Sensor Message)
    - For each camera frame:
        - Timestamp
        - Camera pose in map as 4x4 Matrix [T_map_camera]
        - Camera Image
        - Point cloud in camera frame (as .pcd file, xyzi, where i is distance to mesh in mm)
        - Projected PointCloud in camera image (Only point cloud)
        - Projecctted PointCloud in camera image containing distance of point as grey value
        - Projecctted PointCloud in camera image containing distance to closest mesh as grey value

#### Getting Started
The following extracts all images from cam0 into the output folder <outputFolder>
1. Terminal A: `roscore`
2. Terminal B: `rosparam set use_sim_time true`
3. Terminal B: `roslaunch background_foreground_segmentation dataset_creator_standalone_rviz.launch output_folder:=<outputFolder>/ use_camera_stick:=cam0`
4. Terminal C: Play rosbag `rosbag play --clock <path/to/bagfile>`
5. RVIZ: Align Mesh with Pointcloud and right click marker -> load mesh, publish mesh

Sometimes due to bad timings running the standalone launch script can confuse the state estimator. In this case, start every node seperately:

1. Terminal A: `roscore`
2. Terminal B: `rosparam set use_sim_time true`
3. Terminal B: `roslaunch smb_state_estimator smb_state_estimator_standalone.launch`
4. Terminal C: `roslaunch cpt_selective_icp supermegabot_selective_icp_with_rviz.launch publish_distance:=true`
5. Terminal D: `roslaunch background_foreground_segmentation dataset_creator.launch outputFolder:=<outputFolder>/ use_camera_stick:=cam0`
5. Terminal E: `roslaunch segmentation_filtered_icp extrinsics.launch`
6. Terminal F: `rosbag play --clock <path/to/bagfile>`
5. RVIZ: Align Mesh with Pointcloud and right click marker -> load CAD, publish mesh


### Generating Pseudo Labels on the Fly 
Pseudo Labels can also be generated on the fly for each camera mounted on the robot.
#### For the SMB with a camera stick setup
1. Terminal A: `roslaunch background_foreground_segmentation cla_label_projector_standalone_with_rviz.launch`
2. Terminal B: `rosrun background_foreground_segmentation label_aggregator.py`
3. Terminal C: `rosbag play --clock <path/to/bagfile>`
In order to change the segmentation method, modify the parameters inside the `config/label_aggregator_cla.yaml` file.
   
The labels for each image will be published in the topic defined in the label_aggregator_cla.yaml config file.

## Python Library
### Data Loader
In order to train a segmentation network on pseudo labels generated on the CLA garage or the Viconroom dataset, use the Dataloader Implementation:
```python
dl = DataLoader("<pathToViconOrClaDataset>", 
                 [480, 720],
                    shuffle=True,
                    validationDir="<pathToValidationDataset>",
                    validationMode="<CLA|ARCHE>",
                    batchSize=1,
                    loadDepth=False,
                    trainFilter=None,
                    validationFilter=None,
                    verbose=True)
train_ds, test_ds = dl.getDataset()
```
### Custom Losses
Since the pseudo labels contain 3 instead of 2 classes, losses used to supervise a model with 2 outputs need to be wrapped in an "ignorant" wrapper, ignoring 
the invalid class. The class number 1 is used for the unknown class. 
```python
from bfseg.utils.losses import ignorant_cross_entropy_loss, ignorant_balanced_cross_entropy_loss
# model predictions should be in categorical format
model.compile(loss=ignorant_cross_entropy_loss) # Just mask out invalid classes
model.compile(loss=ignorant_balanced_cross_entropy_loss) # Also remove class inbalance inside labels
```
The Smooth consistency loss and the depth loss can also be importet from the bfseg.utils.losses library:
```python
from bfseg.utils.losses import consistency_loss_from_stacked_prediction, ignorant_depth_loss
```
### Custom Metrics
In order to evaluate a model on the pseudo labels, ignorant metric wrapper have been implemented
```python 
from bfseg.utils.metrics import IgnorantBalancedMeanIoU, IgnorantMeanIoU, IgnorantBalancedAccuracyMetric, IgnorantAccuracyMetric
```
### Train Models
There are 3 different entry points, depending on which model architecture should be trained:
#### SemanticSegmentation
##### Example
```sh 
python background_foreground_segmentation/src/train_experiments/train_sem_seg.py --name_prefix "DEEPLAB_CLA_FUSED_BALANCED" --model_name DEEPLAB  --backbone "xception" --loss_balanced True --train_path "/cluster/scratch/zrene/cla_dataset/fused" --batch_size 4 --output_path "/cluster/scratch/zrene/outputs/results"
```
##### All options
```sh 
python background_foreground_segmentation/src/train_experiments/train_sem_seg.py -h
usage: train_sem_seg.py [-h] [--name_prefix NAME_PREFIX]
                        [--output_path OUTPUT_PATH] [--image_w IMAGE_W]
                        [--image_h IMAGE_H] [--train_path TRAIN_PATH]
                        [--validation_path VALIDATION_PATH]
                        [--validation_mode VALIDATION_MODE]
                        [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
                        [--optimizer_lr OPTIMIZER_LR]
                        [--model_name {PSP,UNET,DEEPLAB}]
                        [--output_stride {8,16}]
                        [--train_from_scratch TRAIN_FROM_SCRATCH]
                        [--loss_balanced LOSS_BALANCED]
                        [--backbone {vgg16,vgg19,resnet18,resnet34,resnet50,resnet101,resnet152,seresnet18,seresnet34,seresnet50,seresnet101,seresnet152,resnext50,resnext101,seresnext50,seresnext101,senet154,densenet121,densenet169,densenet201,inceptionv3,inceptionresnetv2,mobilenet,mobilenetv2,efficientnetb0,efficientnetb1,efficientnetb2,efficientnetb3,efficientnetb4,efficientnetb5, efficientnetb7,xception,mobile}]
                        [--nyu_batchsize NYU_BATCHSIZE] [--nyu_lr NYU_LR]
                        [--nyu_epochs NYU_EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --name_prefix NAME_PREFIX
                        Name Prefix for this experiment (default: )
  --output_path OUTPUT_PATH
                        Output Path, where model will be stored (default: )
  --image_w IMAGE_W     Image width (default: 720)
  --image_h IMAGE_H     Image height (default: 480)
  --train_path TRAIN_PATH
                        Path to dataset (default:
                        /cluster/scratch/zrene/cla_dataset/watershed/)
  --validation_path VALIDATION_PATH
                        Path to validation dataset (default:
                        /cluster/scratch/zrene/cla_dataset/hiveLabels/)
  --validation_mode VALIDATION_MODE
                        Validation Mode <CLA,ARCHE> (default: CLA)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 20)
  --batch_size BATCH_SIZE
                        Number of samples in a batch for training (default: 4)
  --optimizer_lr OPTIMIZER_LR
                        Learning rate at start of training (default: 0.0001)
  --model_name {PSP,UNET,DEEPLAB}
                        CNN architecture (default: PSP)
  --output_stride {8,16}
                        Output stride, only for Deeplab model (default: 16)
  --train_from_scratch TRAIN_FROM_SCRATCH
                        If True, pretrain model on nyu dataset (default:
                        False)
  --loss_balanced LOSS_BALANCED
                        If True, uses balanced losses (default: True)
  --backbone {vgg16,vgg19,resnet18,resnet34,resnet50,resnet101,resnet152,seresnet18,seresnet34,seresnet50,seresnet101,seresnet152,resnext50,resnext101,seresnext50,seresnext101,senet154,densenet121,densenet169,densenet201,inceptionv3,inceptionresnetv2,mobilenet,mobilenetv2,efficientnetb0,efficientnetb1,efficientnetb2,efficientnetb3,efficientnetb4,efficientnetb5, efficientnetb7,xception,mobile}
                        CNN architecture. If using Deeplab model, only
                        xception and mobile are supported (default: vgg16)
  --nyu_batchsize NYU_BATCHSIZE
                        Batchsize to train_experiments on nyu (default: 4)
  --nyu_lr NYU_LR       Learning rate for pretraining on nyu (default: 0.001)
  --nyu_epochs NYU_EPOCHS
                        Number of epochs to train_experiments on nyu (default:
                        20)

```
#### Multi Tasking Architecture (Include Depth prediction)
##### Example
```sh 
python background_foreground_segmentation/src/train_experiments/train_sem_seg_with_depth.py --model_name PSP --name_prefix "PSP_VICON_ONLY_SEMSEG_BALANCED" --loss_balanced True --train_path "/cluster/scratch/zrene/vicon_dataset/rotated" --batch_size 8 --backbone resnet34 --output_path "/cluster/scratch/zrene/outputs/results" --use_consistency_loss False --depth_weigth 4```
```
##### All options
```sh 
python background_foreground_segmentation/src/train_experiments/train_sem_seg.py -h
usage: train_sem_seg_with_depth.py [-h] [--name_prefix NAME_PREFIX]
                                   [--output_path OUTPUT_PATH]
                                   [--train_path TRAIN_PATH]
                                   [--validation_path VALIDATION_PATH]
                                   [--validation_mode VALIDATION_MODE]
                                   [--num_epochs NUM_EPOCHS]
                                   [--batch_size BATCH_SIZE]
                                   [--optimizer_lr OPTIMIZER_LR]
                                   [--output_stride {8,16}]
                                   [--train_from_scratch TRAIN_FROM_SCRATCH]
                                   [--loss_balanced LOSS_BALANCED]
                                   [--backbone {vgg16,vgg19,resnet18,resnet34,resnet50,resnet101,resnet152,seresnet18,seresnet34,seresnet50,seresnet101,seresnet152,resnext50,resnext101,seresnext50,seresnext101,senet154,densenet121,densenet169,densenet201,inceptionv3,inceptionresnetv2,mobilenet,mobilenetv2,efficientnetb0,efficientnetb1,efficientnetb2,efficientnetb3,efficientnetb4,efficientnetb5, efficientnetb7,xception,mobile}]
                                   [--nyu_batchsize NYU_BATCHSIZE]
                                   [--nyu_lr NYU_LR] [--nyu_epochs NYU_EPOCHS]
                                   [--image_w IMAGE_W] [--image_h IMAGE_H]
                                   [--depth_weigth DEPTH_WEIGTH]
                                   [--semseg_weight SEMSEG_WEIGHT]
                                   [--use_consistency_loss USE_CONSISTENCY_LOSS]
                                   [--consistency_weight CONSISTENCY_WEIGHT]
                                   [--model_name MODEL_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --name_prefix NAME_PREFIX
                        Name Prefix for this experiment (default: )
  --output_path OUTPUT_PATH
                        Output Path, where model will be stored (default: )
  --train_path TRAIN_PATH
                        Path to dataset (default:
                        /cluster/scratch/zrene/cla_dataset/watershed/)
  --validation_path VALIDATION_PATH
                        Path to validation dataset (default:
                        /cluster/scratch/zrene/cla_dataset/hiveLabels/)
  --validation_mode VALIDATION_MODE
                        Validation Mode <CLA,ARCHE> (default: CLA)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 20)
  --batch_size BATCH_SIZE
                        Number of samples in a batch for training (default: 4)
  --optimizer_lr OPTIMIZER_LR
                        Learning rate at start of training (default: 0.0001)
  --output_stride {8,16}
                        Output stride, only for Deeplab model (default: 16)
  --train_from_scratch TRAIN_FROM_SCRATCH
                        If True, pretrain model on nyu dataset (default:
                        False)
  --loss_balanced LOSS_BALANCED
                        If True, uses balanced losses (default: True)
  --backbone {vgg16,vgg19,resnet18,resnet34,resnet50,resnet101,resnet152,seresnet18,seresnet34,seresnet50,seresnet101,seresnet152,resnext50,resnext101,seresnext50,seresnext101,senet154,densenet121,densenet169,densenet201,inceptionv3,inceptionresnetv2,mobilenet,mobilenetv2,efficientnetb0,efficientnetb1,efficientnetb2,efficientnetb3,efficientnetb4,efficientnetb5, efficientnetb7,xception,mobile}
                        CNN architecture. If using Deeplab model, only
                        xception and mobile are supported (default: vgg16)
  --nyu_batchsize NYU_BATCHSIZE
                        Batchsize to train_experiments on nyu (default: 4)
  --nyu_lr NYU_LR       Learning rate for pretraining on nyu (default: 0.001)
  --nyu_epochs NYU_EPOCHS
                        Number of epochs to train_experiments on nyu (default:
                        20)
  --image_w IMAGE_W
  --image_h IMAGE_H
  --depth_weigth DEPTH_WEIGTH
  --semseg_weight SEMSEG_WEIGHT
  --use_consistency_loss USE_CONSISTENCY_LOSS
  --consistency_weight CONSISTENCY_WEIGHT
  --model_name MODEL_NAME
```
#### Timestamp Based Training (only train on first T-seconds of trajectory, evaluate on rest)
##### Example
```sh
python background_foreground_segmentation/src/train_experiments/train_sem_seg_for_given_timestamps.py --dataset CLA --train_duration 150 --name_prefix "PSP_CLA_FUSED_BALANCED_150s_pretrained_for_eval" --model_name PSP --num_epochs 10 --backbone "resnet34" --loss_balanced True --train_path "/cluster/scratch/zrene/cla_dataset/fused" --shuffle False --batch_size 1 --output_path "/cluster/scratch/zrene/outputs/results" --image_w 720 --image_h 480
```
##### All options
```sh
usage: train_sem_seg_for_given_timestamps.py [-h] [--name_prefix NAME_PREFIX]
                                             [--output_path OUTPUT_PATH]
                                             [--image_w IMAGE_W]
                                             [--image_h IMAGE_H]
                                             [--train_path TRAIN_PATH]
                                             [--validation_path VALIDATION_PATH]
                                             [--validation_mode VALIDATION_MODE]
                                             [--num_epochs NUM_EPOCHS]
                                             [--batch_size BATCH_SIZE]
                                             [--optimizer_lr OPTIMIZER_LR]
                                             [--model_name {PSP,UNET,DEEPLAB}]
                                             [--output_stride {8,16}]
                                             [--train_from_scratch TRAIN_FROM_SCRATCH]
                                             [--loss_balanced LOSS_BALANCED]
                                             [--backbone {vgg16,vgg19,resnet18,resnet34,resnet50,resnet101,resnet152,seresnet18,seresnet34,seresnet50,seresnet101,seresnet152,resnext50,resnext101,seresnext50,seresnext101,senet154,densenet121,densenet169,densenet201,inceptionv3,inceptionresnetv2,mobilenet,mobilenetv2,efficientnetb0,efficientnetb1,efficientnetb2,efficientnetb3,efficientnetb4,efficientnetb5, efficientnetb7,xception,mobile}]
                                             [--nyu_batchsize NYU_BATCHSIZE]
                                             [--nyu_lr NYU_LR]
                                             [--nyu_epochs NYU_EPOCHS]
                                             [--dataset {CLA,VICON}]
                                             [--routes ROUTES [ROUTES ...]]
                                             [--train_duration TRAIN_DURATION]
                                             [--start_timestamp START_TIMESTAMP]
                                             [--shuffle SHUFFLE]

optional arguments:
  -h, --help            show this help message and exit
  --name_prefix NAME_PREFIX
                        Name Prefix for this experiment (default: )
  --output_path OUTPUT_PATH
                        Output Path, where model will be stored (default: )
  --image_w IMAGE_W     Image width (default: 720)
  --image_h IMAGE_H     Image height (default: 480)
  --train_path TRAIN_PATH
                        Path to dataset (default:
                        /cluster/scratch/zrene/cla_dataset/watershed/)
  --validation_path VALIDATION_PATH
                        Path to validation dataset (default:
                        /cluster/scratch/zrene/cla_dataset/hiveLabels/)
  --validation_mode VALIDATION_MODE
                        Validation Mode <CLA,ARCHE> (default: CLA)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 20)
  --batch_size BATCH_SIZE
                        Number of samples in a batch for training (default: 4)
  --optimizer_lr OPTIMIZER_LR
                        Learning rate at start of training (default: 0.0001)
  --model_name {PSP,UNET,DEEPLAB}
                        CNN architecture (default: PSP)
  --output_stride {8,16}
                        Output stride, only for Deeplab model (default: 16)
  --train_from_scratch TRAIN_FROM_SCRATCH
                        If True, pretrain model on nyu dataset (default:
                        False)
  --loss_balanced LOSS_BALANCED
                        If True, uses balanced losses (default: True)
  --backbone {vgg16,vgg19,resnet18,resnet34,resnet50,resnet101,resnet152,seresnet18,seresnet34,seresnet50,seresnet101,seresnet152,resnext50,resnext101,seresnext50,seresnext101,senet154,densenet121,densenet169,densenet201,inceptionv3,inceptionresnetv2,mobilenet,mobilenetv2,efficientnetb0,efficientnetb1,efficientnetb2,efficientnetb3,efficientnetb4,efficientnetb5, efficientnetb7,xception,mobile}
                        CNN architecture. If using Deeplab model, only
                        xception and mobile are supported (default: vgg16)
  --nyu_batchsize NYU_BATCHSIZE
                        Batchsize to train_experiments on nyu (default: 4)
  --nyu_lr NYU_LR       Learning rate for pretraining on nyu (default: 0.001)
  --nyu_epochs NYU_EPOCHS
                        Number of epochs to train_experiments on nyu (default:
                        20)
  --dataset {CLA,VICON}
  --routes ROUTES [ROUTES ...]
  --train_duration TRAIN_DURATION
                        train_experiments duration in seconds [s] (default:
                        10)
  --start_timestamp START_TIMESTAMP
                        Timestamp where training should start (default: None)
  --shuffle SHUFFLE     Whether to shuffle Images (default: False)

```
