import argparse
from bfseg.utils.utils import str2bool
import segmentation_models as sm

from bfseg.data.meshdist.dataLoader import DataLoader
from bfseg.utils import NyuDataLoader
import os

class SemSegExpirement():
    def __init__(self):
        self.config = self.getConfig();

        if os.environ['local']:
            self.config.train_path = "/home/rene/cla_dataset/watershed/"
            self.config.validation_path = '/home/rene/hiveLabels/'

    def getConfig(self):
        """
        Arguements for this experiment
        """
        parser = argparse.ArgumentParser(
            add_help=True,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self._addArguments(parser)

        return parser.parse_args()

    def _addArguments(self, parser):
        parser.add_argument(
            '--name_prefix', type=str, help='Name Prefix', default="")
        parser.add_argument(
            '--train_path', type=str, help='Path to dataset',
            default="/cluster/scratch/zrene/cla_dataset/watershed/")
        parser.add_argument(
            '--validation_path', type=str, help='Path to dataset',
            default="/cluster/scratch/zrene/cla_dataset/hiveLabels/")
        parser.add_argument(
            '--validation_mode', type=str, help='Validation Mode <CLA,ARCHE>', default="CLA")
        parser.add_argument(
            '--num_epochs', type=int, default=30, help='Number of training epochs')
        parser.add_argument(
            '--batch_size', type=int, default=4, help='Number of samples in a batch for training')
        parser.add_argument(
            '--optimizer_lr', type=float, default=0.001, help='Learning rate at start of training')
        parser.add_argument(
            '--model_name', type=str, default='PSP', choices=['PSP', 'UNET', 'DEEPLAB'], help='CNN architecture')
        parser.add_argument(
            '--backbone', type=str, default='vgg16',
            choices=["vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "seresnet18",
                     "seresnet34", "seresnet50", "seresnet101", "seresnet152", "resnext50", "resnext101",
                     "seresnext50",
                     "seresnext101", "senet154", "densenet121", "densenet169", "densenet201", "inceptionv3",
                     "inceptionresnetv2", "mobilenet", "mobilenetv2", "efficientnetb0", "efficientnetb1",
                     "efficientnetb2",
                     "efficientnetb3", "efficientnetb4", "efficientnetb5", " efficientnetb7"],
            help='CNN architecture')
        parser.add_argument(
            '--baselinePath', type=str, default='./baseline_model.h5')
        parser.add_argument(
            '--train_from_scratch', type=str2bool, default=False)
        parser.add_argument(
            '--loss_balanced', type=str2bool, default=False)
        parser.add_argument(
            '--image_w', type=int, default=720)
        parser.add_argument(
            '--image_h', type=int, default=480)
        parser.add_argument(
            '--nyu_batchsize', type=int, default=4)
        parser.add_argument(
            '--nyu_lr', type=float, default=0.001)
        parser.add_argument(
            '--nyu_epochs', type=int, default=20)



    def getNyuTrainData(self):
        train_ds, train_info, valid_ds, valid_info, test_ds, test_info = NyuDataLoader.NyuDataLoader(
            self.config.nyu_batchsize, (self.config.image_w, self.config.image_h)).getDataSets()

        return train_ds, valids_ds

    def getTrainData(self):
        dl = DataLoader(self.config.train_path, [self.config.image_h, self.config.image_w],
                        validationDir=self.config.validation_path,
                        validationMode=self.config.validation_mode,
                        batchSize=self.config.batch_size)
        return dl.getDataset()

    def getModel(self):
        if self.config.model_name == "PSP":
            model = sm.PSPNet(self.config.backbone, input_shape=(self.config.image_h, self.config.image_w, 3), classes=2)
        elif self.config.model_name == "UNET":
            model = sm.Unet(self.config.backbone, input_shape=(self.config.image_h, self.config.image_w, 3), classes=2)
        elif self.config.model_name == "DEEPLAB":
            from bfseg.models.deeplab import Deeplabv3
            model = Deeplabv3(input_shape=(self.config.image_h, self.config.image_w, 3), classes=2)

        return model

    def compileModel(self, model):
        model.compile(
            loss=ignorant_balanced_cross_entropy_loss if self.config.loss_balanced else ignorant_cross_entropy_loss,
            optimizer=tf.keras.optimizers.Adam(config.optimizer_lr),
            metrics=[IgnorantBalancedAccuracyMetric(),
                     IgnorantAccuracyMetric(),
                     IgnorantMeanIoU(),
                     IgnorantBalancedMeanIoU(),
                     ])

    def compileNyuModel(self, model):
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(self.config.nyu_lr),
            metrics='accuracy')
