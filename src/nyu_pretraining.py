from sacred import Experiment
import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
import os
from shutil import make_archive

import bfseg.data.nyu.Nyu_depth_v2_labeled
from bfseg.utils.losses import IgnorantCrossEntropyLoss
from bfseg.utils.metrics import IgnorantMeanIoU
from bfseg.models.fast_scnn import fast_scnn
from bfseg.utils.utils import crop_map
from bfseg.settings import TMPDIR

from .sacred_utils import get_observer

ex = Experiment()
ex.observers.append(get_observer())


@ex.main
def pretrain_nyu(_run, batchsize=10, learning_rate=1e-4):
  train_data = tfds.load(
      'NyuDepthV2Labeled', split='full[:90%]',
      as_supervised=True).map(crop_map).shuffle(1000).batch(batchsize).cache()
  val_data = tfds.load(
      'NyuDepthV2Labeled', split='full[90%:]',
      as_supervised=True).map(crop_map).batch(batchsize).cache()

  x = tf.keras.Input(shape=train_data.element_spec[0].shape[1:])
  out = tf.image.convert_image_dtype(x, tf.float32)
  out = fast_scnn(out, num_downsampling_layers=1, num_classes=2)
  model = tf.keras.Model(inputs=x, outputs=out)
  model.compile(loss=IgnorantCrossEntropyLoss(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                metrics=[IgnorantMeanIoU()])
  history = model.fit(train_data, epochs=2, validation_data=val_data)
  model.save(os.path.join(TMPDIR, 'model'))
  make_archive(os.path.join(TMPDIR, 'model.zip'), 'zip',
               os.path.join(TMPDIR, 'model'))
  _run.add_artifact(os.path.join(TMPDIR, 'model.zip'))
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  _run.info['final_loss'] = hist['loss'].iloc[-1]
  _run.info['final miou'] = hist['val_ignorant_mean_io_u'].iloc[-1]
  for _, row in hist.iterrows():
    _run.log_scalar('training.loss', row['loss'], row['epoch'])
    _run.log_scalar('validation.loss', row['val_loss'], row['epoch'])


if __name__ == '__main__':
  ex.run_commandline()
