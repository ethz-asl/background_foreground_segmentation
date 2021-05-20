import tensorflow_datasets as tfds

import bfseg.data.nyu.Nyu_depth_v2_labeled
import bfseg.data.hive.bfseg_validation_labeled
import bfseg.data.hive.office_rumlang_validation_labeled
import bfseg.data.meshdist.bfseg_cla_meshdist_labels
import bfseg.data.pseudolabels

tfds.load('OfficeRumlangValidationLabeled')
tfds.load('MeshdistPseudolabels')
tfds.load('NyuDepthV2Labeled')
tfds.load("BfsegValidationLabeled")
tfds.load("meshdist_pseudolabels")
tfds.load("office_rumlang_validation_labeled")
tfds.load('BfsegCLAMeshdistLabels')

