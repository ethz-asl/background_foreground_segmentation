###########################################################################################
#    Main entry point to train model
###########################################################################################

# disable GPU if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from bfseg.data.meshdist.dataLoader import DataLoader
from bfseg.utils.losses import ignorant_cross_entropy_loss
from bfseg.utils.tbMonitor import TensorBoardMonitor
import segmentation_models as sm
import tensorflow as tf
from bfseg.utils.metrics import IgnorantBalancedAccuracyMetric

from sacred import Experiment
from sacred.observers import MongoObserver

import loadBaselineData as BaselineData

ex = Experiment('Self_Supervised_BF_seg')

ex.observers.append(
    MongoObserver(
        url=
        'mongodb://bfseg_runner:y5jL6uTnHN3W4AZo5oCiG3iX@data.asl.ethz.ch/bfseg',
        db_name='bfseg'))

# Tweak GPU settings for local use
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)


@ex.config
def dnn_config():
  input_dim = 100
  output_dim = 20
  neurons = 64
  activation = 'relu'
  dropout = 0.4


def pretrainOnNyu(model):
  batchSize = 4
  train_ds, train_info, valid_ds, valid_info, test_ds, test_info = BaselineData.getDataSets(
      batchSize)

  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.Adam(0.001),
      metrics='accuracy',
  )
  callbacks = [
      tf.keras.callbacks.ModelCheckpoint('./best_model.h5',
                                         save_weights_only=True,
                                         save_best_only=True,
                                         mode='min'),
      tf.keras.callbacks.ReduceLROnPlateau(),
  ]

  model.fit(
      train_ds,
      # steps_per_epoch=int(train_info.splits['train'].num_examples*0.8)//batch_size,
      steps_per_epoch=train_info.splits['train'].num_examples // batchSize,
      epochs=10,
      validation_data=test_ds,
      callbacks=callbacks,
  )


@ex.automain
def main(_run):
  # workingdir = "/home/rene/cla_dataset/watershed/"
  workingdir = "/cluster/scratch/zrene/cla_dataset/watershed/"
  # validationDir = '/home/rene/hiveLabels/'
  validationDir = '/cluster/scratch/zrene/cla_dataset/hiveLabels/'
  summary = "watershed_30ep_augmentation_PSPNET_vgg16"

  # Desired image shape. Input images will be cropped + scaled to this shape
  image_w = 720
  image_h = 480
  trainFromScratch = True

  dataLoader = DataLoader(workingdir, [image_h, image_w],
                          validationDir=validationDir,
                          validationMode="CLA",
                          batchSize=4)
  train_ds, test_ds = dataLoader.getDataset()

  BACKBONE = "vgg16"
  model = sm.PSPNet(BACKBONE, input_shape=(image_h, image_w, 3), classes=2)

  if trainFromScratch:
    pretrainOnNyu(model)
    return

  model.compile(loss=ignorant_cross_entropy_loss,
                optimizer=tf.keras.optimizers.Adam(0.01),
                metrics=[
                    IgnorantBalancedAccuracyMetric(),
                ])

  # Training callbacks
  callbacks = []
  # Set up tensorboard to monitor progess
  monitor = TensorBoardMonitor(train_ds, test_ds, model, tag=summary)
  monitor.startTensorboard()
  callbacks = [
      tf.keras.callbacks.ModelCheckpoint(
          './model.{epoch:02d}-{val_loss:.2f}.h5',
          save_weights_only=True,
          save_best_only=False,
          mode='min'),
      tf.keras.callbacks.ReduceLROnPlateau(),
  ]

  callbacks.extend(monitor.getCallbacks())

  history = model.fit(
      train_ds,
      #steps_per_epoch = 5,
      epochs=30,
      validation_data=test_ds,
      callbacks=callbacks)

  # Save validation loss or other metric in sacred
  for idx, loss in enumerate(history.history['val_loss']):
    _run.log_scalar("validation.loss", loss, idx)


if __name__ == "__main__":
  sm.set_framework('tf.keras')
  main()
  # input("Press enter to stop")
