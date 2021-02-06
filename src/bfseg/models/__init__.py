# Necessary for `segmentation_models`.
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from .fast_scnn import fast_scnn as FastSCNN
from segmentation_models import Unet as UNet