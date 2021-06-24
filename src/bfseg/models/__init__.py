# Necessary for `segmentation_models`.
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from .fast_scnn import fast_scnn as FastSCNN
from .fast_scnn_plus_depth import fast_scnn_plus_depth as FastSCNNPlusDepth
from .fast_scnn_plus_dorn import fast_scnn_plus_dorn as FastSCNNPlusDorn
from segmentation_models import Unet as UNet