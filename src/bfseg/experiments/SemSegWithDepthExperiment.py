import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
import argparse
import segmentation_models as sm

from bfseg.utils.utils import str2bool
from bfseg.utils import NyuDataLoader
from bfseg.utils.metrics import IgnorantBalancedAccuracyMetric, IgnorantAccuracyMetric, IgnorantBalancedMeanIoU, \
    IgnorantMeanIoU, IgnorantDepthMAPE
from bfseg.utils.losses import ignorant_cross_entropy_loss, ignorant_balanced_cross_entropy_loss
from bfseg.data.meshdist.dataLoader import DataLoader

from bfseg.experiments.SemSegExperiment import SemSegExperiment
import bfseg.models.MultiTaskingModels as mtm
from bfseg.utils.NyuDataLoader import NyuDataLoader
from bfseg.utils.losses import ignorant_balanced_cross_entropy_loss,ignorant_depth_loss

from bfseg.utils.evaluation import scoreAndPlotPredictions

class SemSegWithDepthExperiment(SemSegExperiment):
  """ Experiment to train ForegroundBackground Semantic Segmentation on meshdist train data """

  def __init__(self):
    super(SemSegWithDepthExperiment, self).__init__()
    # Get a dataloader to load training images
    # Get a dataloader to load training images
    self.dl = DataLoader(self.config.train_path,
                         [self.config.image_h, self.config.image_w],
                         validationDir=self.config.validation_path,
                         validationMode=self.config.validation_mode,
                         batchSize=self.config.batch_size, loadDepth=True)

    self.nyuLoader = NyuDataLoader( self.config.nyu_batchsize, (self.config.image_w, self.config.image_h), loadDepth = True)

    self.numTestImages = self.dl.validationSize



  def _addArguments(self, parser):
    """ Add custom arguments that are needed for this experiment """
    super(SemSegWithDepthExperiment, self)._addArguments(parser)

    # Change default image size since we are now using the kinect
    parser.add_argument('--image_w', type=int, default=480)
    parser.add_argument('--image_h', type=int, default=480)

  def getNyuTrainData(self):
    """ Return training data from NYU. In order to scale the images to the right format, a custom dataloader
            with map function was implemented """
    train_ds, train_info, valid_ds, valid_info, test_ds, test_info = self.nyuLoader.getDataSets()

    steps_per_epoch = train_info.splits[
        'train'].num_examples // self.config.nyu_batchsize
    return train_ds, valid_ds, steps_per_epoch

  def getTrainData(self):
    """ return train_ds, test_ds """
    return self.dl.getDataset()

  def getModel(self):
    if self.config.model_name == "PSP":
      print("buliding PSP model")
      #input = tf.keras.layers.Input(shape = (self.config.image_h, self.config.image_w, 3), name="image")
      model = mtm.PSPNet(self.config.backbone,
                        input_shape=(self.config.image_h, self.config.image_w,
                                     3),
                        classes=2)
      model.summary()
      return model
    return mtm.Deeplabv3(input_shape=(self.config.image_h, self.config.image_w,  3),classes=2, activation="sigmoid")
    #
    # elif self.config.model_name == "DEEPLAB":
    #   from bfseg.models.deeplab import Deeplabv3
    #   model = Deeplabv3(input_shape=(self.config.image_h, self.config.image_w,
    #                                  3),
    #                     classes=2)
    #
    # inp = tf.keras.layers.Input(shape = (self.config.image_h, self.config.image_w, 3))
    # out1 = tf.keras.backend.argmax(model(inp), axis = -1)
    # out2 = model(inp)
    #
    # return tf.keras.models.Model(inputs = inp, outputs= [out1, out2])

  def compileModel(self, model):
      model.compile(loss = [ignorant_depth_loss, ignorant_balanced_cross_entropy_loss],
                    optimizer=tf.keras.optimizers.Adam(self.config.optimizer_lr),
                    metrics={'depth': [IgnorantDepthMAPE()],
                             'semseg': [IgnorantBalancedAccuracyMetric(), IgnorantAccuracyMetric(), IgnorantMeanIoU(),
                                        IgnorantBalancedMeanIoU()]}
      )
      # model.useIgnorantLosses = True
      # super(SemSegWithDepthExperiment, self).compileModel(model)
      #   model.compile(loss=self.getLoss(),
      #                 optimizer=tf.keras.optimizers.Adam(self.config.optimizer_lr),
      #                 metrics=self.getMetrics())

  def compileNyuModel(self, model):
    model.useIgnorantLosses = False
    model.compile(loss = [ignorant_depth_loss, ignorant_balanced_cross_entropy_loss],  optimizer=tf.keras.optimizers.Adam(self.config.nyu_lr), metrics = {'depth': [IgnorantDepthMAPE()], 'semseg': [IgnorantBalancedAccuracyMetric(), IgnorantAccuracyMetric(), IgnorantMeanIoU(), IgnorantBalancedMeanIoU()]})

  def loss(self):
      def l(*args, **kwargs):
          print(args)
          print(kwargs)
          return 0#tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
      return l

  def lossMSE(self):
      def l(y_true, y_pred):
          print(y_pred, "mse")
          print(y_true, "mse")
          return tf.keras.losses.mean_squared_logarithmic_error(y_true, y_pred)

      return l
  def getLoss(self):
    return ignorant_balanced_cross_entropy_loss if self.config.loss_balanced else ignorant_cross_entropy_loss

  def getMetrics(self):
    return [
        IgnorantBalancedAccuracyMetric(),
        IgnorantAccuracyMetric(),
        IgnorantMeanIoU(),
        IgnorantBalancedMeanIoU(),
    ]

  def scoreModel(self, model, test_ds):
      scoreAndPlotPredictions(lambda img: model.predict(img)[1],
                              test_ds,
                              self.numTestImages,
                              plot=False)