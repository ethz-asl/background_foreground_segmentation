###########################################################################################
#    Main entry point to train model
###########################################################################################

# disable GPU if needed
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["SM_FRAMEWORK"] = "tf.keras"

from bfseg.data.meshdist.dataLoader import DataLoader
from bfseg.utils.losses import ignorant_cross_entropy_loss
import segmentation_models as sm
import tensorflow as tf
from bfseg.utils.metrics import IgnorantBalancedAccuracyMetric, IgnorantAccuracyMetric, IgnorantBalancedMeanIoU, IgnorantMeanIoU

from bfseg.utils import NyuDataLoader
from sacred import Experiment
from sacred.observers import MongoObserver
import argparse

import argparse
import json


# TODO move to function
parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

parser.add_argument(
    '--name_prefix', type=str, help='Name Prefix', default = "")
parser.add_argument(
    '--train_path', type=str, help='Path to dataset', default = "/cluster/scratch/zrene/cla_dataset/watershed/")
parser.add_argument(
    '--validation_path', type=str, help='Path to dataset', default ="/cluster/scratch/zrene/cla_dataset/hiveLabels/")
parser.add_argument(
    '--validation_mode', type=str, help='Validation Mode <CLA,ARCHE>', default = "CLA")
parser.add_argument(
    '--num_epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument(
    '--batch_size', type=int, default=4, help='Number of samples in a batch for training')
parser.add_argument(
    '--optimizer_lr', type=float, default=0.001, help='Learning rate at start of training')
parser.add_argument(
    '--model_name', type=str, default='PSP', choices=['PSP'], help='CNN architecture')
parser.add_argument(
    '--backbone', type=str, default='vgg16', choices=["vgg16","vgg19","resnet18","resnet34","resnet50","resnet101","resnet152","seresnet18","seresnet34","seresnet50","seresnet101","seresnet152","resnext50","resnext101","seresnext50","seresnext101","senet154","densenet121","densenet169","densenet201","inceptionv3","inceptionresnetv2","mobilenet","mobilenetv2","efficientnetb0","efficientnetb1","efficientnetb2","efficientnetb3","efficientnetb4","efficientnetb5"," efficientnetb7"], help='CNN architecture')

args = parser.parse_args()
experiment_name = "{}_{}_{}_{}lr_{}bs_{}ep".format(args.name_prefix,args.backbone, args.model_name, args.optimizer_lr, args.batch_size, args.num_epochs)

print(experiment_name)
print(json.dumps(args.__dict__, indent=4, sort_keys=True))

ex = Experiment(experiment_name)

ex.observers.append(
    MongoObserver(
        url=
        'mongodb://bfseg_runner:y5jL6uTnHN3W4AZo5oCiG3iX@data.asl.ethz.ch/bfseg',
        db_name='bfseg'))

# Tweak GPU settings for local use
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# workingdir = "/cluster/scratch/zrene/cla_dataset/watershed/"
# validationDir = '/cluster/scratch/zrene/cla_dataset/hiveLabels/'
# baselinePath = "./baseline_model.h5"
#
# try:
#   if os.environ['local']:
#     workingdir = "/home/rene/cla_dataset/watershed/"
#     validationDir = '/home/rene/hiveLabels/'
# except:
#   print("Running on cluster")

# Desired image shape. Input images will be cropped + scaled to this shape
image_w = 720
image_h = 480


class LogMetrics(tf.keras.callbacks.Callback):
  """
  Logs the metrics of the current epoch in tensorboard
  """

  def on_epoch_end(self, _, logs={}):
    my_metrics(logs=logs)


@ex.capture
def my_metrics(_run, logs):
  for key, value in logs.items():
    _run.log_scalar(key, float(value))


@ex.config
def cfg():
  config = args
  config.image_size = (image_w, image_h)


def pretrainOnNyu(model, batchSize=4, epochs=10):
  """
  Pretrain this model on the nyu dataset
  """
  train_ds, train_info, valid_ds, valid_info, test_ds, test_info = NyuDataLoader.NyuDataLoader(
      batchSize, (image_w, image_h)).getDataSets()

  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.Adam(0.001),
      metrics='accuracy',
  )

  callbacks = [
      tf.keras.callbacks.ModelCheckpoint(baselinePath,
                                         save_weights_only=True,
                                         save_best_only=True,
                                         mode='min'),
      tf.keras.callbacks.ReduceLROnPlateau(),
      LogMetrics()
  ]

  model.fit(
      train_ds,
      # steps_per_epoch=int(train_info.splits['train'].num_examples*0.8)//batch_size,
      steps_per_epoch=train_info.splits['train'].num_examples // batchSize,
      epochs=epochs,
      validation_data=test_ds,
      callbacks=callbacks,
  )


@ex.automain
def run(config):
  # Do not use pretrained weights but generate new ones by training on nyu
  trainFromScratch = False

  model = sm.PSPNet("vgg16", input_shape=(image_h, image_w, 3), classes=2)

  if trainFromScratch:
    # pretrain model on Nyu dataset
    pretrainOnNyu(model)
  else:
    try:
      model.load_weights(baselinePath)
    except:
      print(
          "Could not load model weights. Starting with random initialized model"
      )

  model.compile(
      loss=ignorant_cross_entropy_loss,
      optimizer=tf.keras.optimizers.Adam(config.optimizer_lr),
      metrics=[IgnorantBalancedAccuracyMetric(),
               IgnorantAccuracyMetric(),
               IgnorantMeanIoU(),
               IgnorantBalancedMeanIoU(),
            ])
  train_ds, test_ds = DataLoader(config.train_path, [image_h, image_w],
                                 validationDir=config.validation_path,
                                 validationMode="CLA",
                                 batchSize=config.batch_size).getDataset()
  # Training callbacks
  # callbacks = []
  # Set up tensorboard to monitor progess
  # monitor = TensorBoardMonitor(train_ds, test_ds, model, tag=summary)
  # monitor.startTensorboard()

  callbacks = [
      tf.keras.callbacks.ModelCheckpoint(
          './model.{epoch:02d}-{val_loss:.2f}.h5',
          save_weights_only=True,
          save_best_only=False,
          mode='min'),
      tf.keras.callbacks.ReduceLROnPlateau(),
      # Log metrics to sacred
      LogMetrics()
  ]

  # callbacks.extend(monitor.getCallbacks())

  model.fit(train_ds,
            epochs=config.num_epochs,
            validation_data=test_ds,
            callbacks=callbacks)


# if __name__ == "__main__":
#   run()
# input("Press enter to stop")