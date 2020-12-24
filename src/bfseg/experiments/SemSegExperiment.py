import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
import argparse
import segmentation_models as sm

from bfseg.utils.utils import str2bool
from bfseg.utils import NyuDataLoader
from bfseg.utils.metrics import IgnorantBalancedAccuracyMetric, IgnorantAccuracyMetric, IgnorantBalancedMeanIoU, \
    IgnorantMeanIoU
from bfseg.utils.losses import ignorant_cross_entropy_loss, ignorant_balanced_cross_entropy_loss
from bfseg.data.meshdist.dataLoader import DataLoader
from bfseg.experiments.Experiment import Experiment


class SemSegExperiment(Experiment):
  """ Experiment to train ForegroundBackground Semantic Segmentation on meshdist train data """

  def __init__(self):
    super(SemSegExperiment, self).__init__()

    # Get a dataloader to load training images
    self.dl = DataLoader(self.config.train_path,
                         [self.config.image_h, self.config.image_w],
                         validationDir=self.config.validation_path,
                         validationMode=self.config.validation_mode,
                         batchSize=self.config.batch_size,
                         loadDepth=False)

    self.nyuLoader = NyuDataLoader.NyuDataLoader(
        self.config.nyu_batchsize, (self.config.image_w, self.config.image_h),
        loadDepth=False)

    self.numTestImages = self.dl.validationSize

  def _addArguments(self, parser):
    """ Add custom arguments that are needed for this experiment """
    super(SemSegExperiment, self)._addArguments(parser)

    parser.add_argument('--train_path',
                        type=str,
                        help='Path to dataset',
                        default="/cluster/scratch/zrene/cla_dataset/watershed/")
    parser.add_argument(
        '--validation_path',
        type=str,
        help='Path to dataset',
        default="/cluster/scratch/zrene/cla_dataset/hiveLabels/")
    parser.add_argument('--validation_mode',
                        type=str,
                        help='Validation Mode <CLA,ARCHE>',
                        default="CLA")
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Number of samples in a batch for training')
    parser.add_argument('--optimizer_lr',
                        type=float,
                        default=0.0001,
                        help='Learning rate at start of training')
    parser.add_argument('--model_name',
                        type=str,
                        default='PSP',
                        choices=['PSP', 'UNET', 'DEEPLAB'],
                        help='CNN architecture')
    parser.add_argument('--output_stride',
                        type=int,
                        default=16,
                        help='Output stride, only for Deeplab model')
    parser.add_argument('--baselinePath',
                        type=str,
                        default='./baseline_model.h5')
    parser.add_argument('--train_from_scratch',
                        type=str2bool,
                        default=False,
                        help="If True, pretrain model on nyu dataset")
    parser.add_argument('--loss_balanced',
                        type=str2bool,
                        default=True,
                        help="If True, uses balanced losses (semseg)")
    parser.add_argument('--image_w', type=int, default=720, help="Image width")
    parser.add_argument('--image_h', type=int, default=480, help="Image height")
    parser.add_argument(
        '--backbone',
        type=str,
        default='vgg16',
        choices=[
            "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "resnet101",
            "resnet152", "seresnet18", "seresnet34", "seresnet50",
            "seresnet101", "seresnet152", "resnext50", "resnext101",
            "seresnext50", "seresnext101", "senet154", "densenet121",
            "densenet169", "densenet201", "inceptionv3", "inceptionresnetv2",
            "mobilenet", "mobilenetv2", "efficientnetb0", "efficientnetb1",
            "efficientnetb2", "efficientnetb3", "efficientnetb4",
            "efficientnetb5", " efficientnetb7"
        ],
        help='CNN architecture')

    # NYU parameters
    parser.add_argument('--nyu_batchsize',
                        type=int,
                        default=4,
                        help="Batchsize to train on nyu")
    parser.add_argument('--nyu_lr',
                        type=float,
                        default=0.001,
                        help="Learning rate for pretraining on nyu")
    parser.add_argument('--nyu_epochs',
                        type=int,
                        default=20,
                        help="Number of epochs to train on nyu")

  def getNyuTrainData(self):
    """ Return training data from NYU. In order to scale the images to the right format, a custom dataloader
            with map function was implemented """
    train_ds, train_info, valid_ds, valid_info, test_ds, test_info = self.nyuLoader.getDataSets(
    )

    steps_per_epoch = train_info.splits[
        'train'].num_examples // self.config.nyu_batchsize
    return train_ds, valid_ds, steps_per_epoch

  def getTrainData(self):
    """ return train_ds, test_ds """
    return self.dl.getDataset()

  def getModel(self):
    if self.config.model_name == "PSP":
      model = sm.PSPNet(self.config.backbone,
                        input_shape=(self.config.image_h, self.config.image_w,
                                     3),
                        classes=2)
    elif self.config.model_name == "UNET":
      model = sm.Unet(self.config.backbone,
                      input_shape=(self.config.image_h, self.config.image_w, 3),
                      classes=2)

    elif self.config.model_name == "DEEPLAB":
      from bfseg.models.deeplab import Deeplabv3
      model = Deeplabv3(input_shape=(self.config.image_h, self.config.image_w,
                                     3),
                        classes=2)

    return model

  def compileModel(self, model):
    model.compile(loss=self.getLoss(),
                  optimizer=tf.keras.optimizers.Adam(self.config.optimizer_lr),
                  metrics=self.getMetrics())

  def compileNyuModel(self, model):
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(self.config.nyu_lr),
                  metrics='accuracy')

  def getLoss(self):
    return ignorant_balanced_cross_entropy_loss if self.config.loss_balanced else ignorant_cross_entropy_loss

  def getMetrics(self):
    return [
        IgnorantBalancedAccuracyMetric(),
        IgnorantAccuracyMetric(),
        IgnorantMeanIoU(),
        IgnorantBalancedMeanIoU(),
    ]
