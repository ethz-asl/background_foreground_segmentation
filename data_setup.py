import tensorflow_datasets as tfds

import bfseg.data.nyu.Nyu_depth_v2_labeled
import bfseg.data.nyu_subsampled
import bfseg.data.hive.bfseg_validation_labeled
import bfseg.data.hive.office_rumlang_validation_labeled
import bfseg.data.meshdist.bfseg_cla_meshdist_labels
import bfseg.data.pseudolabels

tfds.load('OfficeRumlangValidationLabeled')
tfds.load('MeshdistPseudolabels')
tfds.load('NyuDepthV2Labeled')
tfds.load('NyuDepthV2Labeled', split='full')
tfds.load('nyu_subsampled')
tfds.load("BfsegValidationLabeled")
tfds.load("meshdist_pseudolabels")
tfds.load("meshdist_pseudolabels", split='office6-2')
tfds.load("office_rumlang_validation_labeled")
tfds.load('BfsegCLAMeshdistLabels')
