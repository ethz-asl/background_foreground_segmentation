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
from bfseg.utils.NyuDataLoader import NyuDataLoader
from bfseg.utils.losses import ignorant_balanced_cross_entropy_loss, ignorant_depth_loss, depth_loss_function
from bfseg.utils.losses import smooth_consistency_loss
from bfseg.utils.evaluation import scoreAndPlotPredictions
from bfseg.models.DeeplaabV3Plus import Deeplabv3


class SemSegWithDepthExperiment(SemSegExperiment):
  """ Experiment to train ForegroundBackground Semantic Segmentation on meshdist train data using additional depth labels"""

  def __init__(self):
    super(SemSegWithDepthExperiment, self).__init__()
    # Get a dataloader to load training images
    self.dl = DataLoader(self.config.train_path,
                         [self.config.image_h, self.config.image_w],
                         validationDir=self.config.validation_path,
                         validationMode="CLA",
                         batchSize=self.config.batch_size,
                         loadDepth=True,
                         cropOptions={
                             'top': 0,
                             'bottom': 0
                         })

    # Get a dataloader to load training images
    self.dl_arche = DataLoader(self.config.train_path,
                               [self.config.image_h, self.config.image_w],
                               validationDir=self.config.validation_path,
                               validationMode="ARCHE",
                               batchSize=self.config.batch_size,
                               loadDepth=True,
                               cropOptions={
                                   'top': 0,
                                   'bottom': 0
                               })

    self.nyuLoader = NyuDataLoader(self.config.nyu_batchsize,
                                   (self.config.image_w, self.config.image_h),
                                   loadDepth=True)

    self.numTestImages = self.dl.validationSize

    self.weightsFolder = self.weightsFolder + "_with_depth"

  def _addArguments(self, parser):
    """ Add custom arguments that are needed for this experiment """
    super(SemSegWithDepthExperiment, self)._addArguments(parser)

    # Change default image size since we are now using the kinect
    parser.add_argument('--image_w', type=int, default=640)
    parser.add_argument('--image_h', type=int, default=480)
    parser.add_argument('--depth_weigth', type=float, default=4)
    parser.add_argument('--semseg_weight', type=float, default=1)
    parser.add_argument('--use_consistency_loss', type=str2bool, default=False)
    parser.add_argument('--consistency_weight', type=float, default=10)
    parser.add_argument('--model_name', type=str, default="DEEPLAB")

    parser.add_argument('--backbone',
                        type=str,
                        default='xception',
                        choices=["xception", "mobile"],
                        help='CNN architecture')

  def getNyuTrainData(self):
    """ Return training data from NYU. In order to scale the images to the right format, a custom dataloader
                with map function was implemented """
    train_ds, train_info, valid_ds, _, _, _ = self.nyuLoader.getDataSets()

    steps_per_epoch = train_info.splits[
        'train'].num_examples // self.config.nyu_batchsize
    return train_ds, valid_ds, steps_per_epoch

  def getTrainData(self):
    """ return train_ds, test_ds """
    return self.dl.getDataset()

  def getModel(self):
    if self.config.model_name != "DEEPLAB":
      raise ValueError(
          "Only DEEPLAB is supported as model for depth prediction")

    model = Deeplabv3(input_shape=(self.config.image_h, self.config.image_w, 3),
                      classes=2,
                      OS=self.config.output_stride,
                      activation="sigmoid",
                      add_depth_prediction=True)

    return model

  def compileModel(self, model):
    loss_weights = {
        'depth': self.config.depth_weigth,
        'semseg': self.config.semseg_weight
    }

    if self.config.use_consistency_loss:
      loss_weights['combined'] = self.config.consistency_weight

    model.compile(loss=self.getLoss(),
                  loss_weights=loss_weights,
                  optimizer=tf.keras.optimizers.Adam(self.config.optimizer_lr),
                  metrics=self.getMetrics())

  def compileNyuModel(self, model):
    model.useIgnorantLosses = False
    model.compile(loss={
        'depth': depth_loss_function,
        'semseg': tf.keras.losses.sparse_categorical_crossentropy
    },
                  optimizer=tf.keras.optimizers.Adam(self.config.nyu_lr),
                  metrics=[
                      tf.keras.metrics.MeanAbsoluteError(),
                      tf.keras.metrics.SparseCategoricalAccuracy()
                  ])

  def consistency_loss(self):

    def loss(y_true=None, y_pred=None):
      depth_pred = y_pred[..., 0]
      semseg_pred = tf.argmax(tf.gather(y_pred, tf.constant([1, 2]), axis=-1),
                              axis=-1)
      consistency_loss = smooth_consistency_loss(depth_pred, semseg_pred,
                                                 0) + smooth_consistency_loss(
                                                     depth_pred, semseg_pred, 1)
      return consistency_loss

    return loss

  def getLoss(self):
    losses = {
        'depth':
            ignorant_depth_loss,
        'semseg':
            ignorant_balanced_cross_entropy_loss
            if self.config.loss_balanced else ignorant_cross_entropy_loss()
    }

    if self.config.use_consistency_loss:
      losses['combined'] = self.consistency_loss()

    return losses

  def getMetrics(self):
    return {
        'depth': [IgnorantDepthMAPE()],
        'semseg': [
            IgnorantBalancedAccuracyMetric(),
            IgnorantAccuracyMetric(),
            IgnorantMeanIoU(),
            IgnorantBalancedMeanIoU()
        ]
    }

  def scoreModel(self, model, outFolder=None, exportImages=False, tag=""):
    print("=========== Evaluating Model on CLA =========")
    model.summary()
    scoreAndPlotPredictions(lambda img: model.predict(img)[1],
                            self.dl.getValidationDataset(),
                            self.dl.validationSize,
                            plot=False,
                            batchSize=self.config.batch_size,
                            outFolder=outFolder,
                            tag=tag + "CLA",
                            exportPredictions=exportImages)

    print("=========== Evaluating Model on ARCHE ===========")
    scoreAndPlotPredictions(lambda img: model.predict(img)[1],
                            self.dl.getValidationDataset(),
                            self.dl_arche.validationSize,
                            plot=False,
                            batchSize=self.config.batch_size,
                            outFolder=outFolder,
                            tag=tag + "ARCHE",
                            exportPredictions=exportImages)
