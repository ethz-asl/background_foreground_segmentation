###########################################################################################
#    Main entry point to train model
###########################################################################################

# disable GPU if needed
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from bfseg.data.meshdist.dataLoader import DataLoader
from bfseg.utils.losses import ignorant_cross_entropy_loss
import segmentation_models as sm
import tensorflow as tf
from bfseg.utils.metrics import IgnorantBalancedAccuracyMetric, IgnorantAccuracyMetric

from bfseg.utils import NyuDataLoader
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('Watershed_pretrained_ep_25_lr_0_01_VGG_PSP')

ex.observers.append(
    MongoObserver(
        url=
        'mongodb://bfseg_runner:y5jL6uTnHN3W4AZo5oCiG3iX@data.asl.ethz.ch/bfseg',
        db_name='bfseg'))

# Tweak GPU settings for local use
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

workingdir = "/cluster/scratch/zrene/cla_dataset/watershed/"
validationDir = '/cluster/scratch/zrene/cla_dataset/hiveLabels/'
baselinePath = "././baseline_model.h5"

try:
  if os.environ['local']:
    workingdir = "/home/rene/cla_dataset/watershed/"
    validationDir = '/home/rene/hiveLabels/'
except:
  print("Running on cluster")

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
  batch = 4
  number_epochs = 25
  image_size = (image_w, image_h)
  model_loss = 'categorical_crossentropy'
  optimizer = 'Adam'
  learning_rate = 0.01
  model = "VGG16_PSP"
  summary = "watershed_30ep_augmentation_PSPNET_vgg16"


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
def run(batch, number_epochs, learning_rate, summary):
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
      optimizer=tf.keras.optimizers.Adam(learning_rate),
      metrics=[IgnorantBalancedAccuracyMetric(),
               IgnorantAccuracyMetric()])

  train_ds, test_ds = DataLoader(workingdir, [image_h, image_w],
                                 validationDir=validationDir,
                                 validationMode="CLA",
                                 batchSize=batch).getDataset()
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
            epochs=number_epochs,
            validation_data=test_ds,
            callbacks=callbacks)


# if __name__ == "__main__":
#   run()
# input("Press enter to stop")
