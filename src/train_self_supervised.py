###########################################################################################
#    Main entry point to train model
###########################################################################################
import os

# disable GPU if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from bfseg.data.meshdist.dataLoader import DataLoader
from bfseg.utils.losses import  ignorant_cross_entropy_loss
from bfseg.utils.tbMonitor import TensorBoardMonitor
import segmentation_models as sm
import tensorflow as tf

# Tweak GPU settings for local use
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def main():
  workingdir = "/home/rene/cla_dataset/watershed/"

  # Desired image shape. Input images will be cropped + scaled to this shape
  image_w = 720
  image_h = 480

  dataLoader = DataLoader(workingdir, [image_h, image_w],
                          validationDir='/home/rene/hiveLabels/',
                          validationMode="CLA")
  train_ds, test_ds = dataLoader.getDataset()

  BACKBONE = "vgg16"
  model = sm.PSPNet(BACKBONE, input_shape=(image_h, image_w, 3), classes=2)

  model.compile(loss=ignorant_cross_entropy_loss,
                optimizer=tf.keras.optimizers.Adam(0.01),
                metrics=['accuracy'])

  # Training callbacks
  callbacks = []
  # Set up tensorboard to monitor progess
  monitor = TensorBoardMonitor(train_ds, test_ds, model)
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

  model.fit(
      train_ds,
      #steps_per_epoch = 5,
      epochs=15,
      validation_data=test_ds,
      callbacks=callbacks)


if __name__ == "__main__":
  sm.set_framework('tf.keras')
  main()
  input("Press enter to stop")
