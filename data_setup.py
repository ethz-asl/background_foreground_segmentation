import tensorflow_datasets as tfds

import bfseg.data.nyu.Nyu_depth_v2_labeled
import bfseg.data.hive.bfseg_validation_labeled

tfds.load('NyuDepthV2Labeled')
tfds.load("BfsegValidationLabeled")
