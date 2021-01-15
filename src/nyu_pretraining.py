from sacred import Experiment
import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
import os
from shutil import make_archive

import bfseg.data.nyu.Nyu_depth_v2_labeled
from bfseg.utils.metrics import IgnorantMeanIoU
from bfseg.models.fast_scnn import fast_scnn
from bfseg.utils.utils import crop_map
from bfseg.settings import TMPDIR
from bfseg.sacred_utils import get_observer

ex = Experiment()
ex.observers.append(get_observer())


@ex.main
def pretrain_nyu(_run, batchsize=10, epochs=100, learning_rate=1e-4, test=False):
  train_data = tfds.load(
      'NyuDepthV2Labeled', split='full[:90%]',
      as_supervised=True).map(crop_map).shuffle(1000).batch(batchsize).cache()
  val_data = tfds.load(
      'NyuDepthV2Labeled', split='full[90%:]',
      as_supervised=True).map(crop_map).batch(batchsize).cache()
  if test:
    # for testing
    train_data = train_data.take(1)
    val_data = val_data.take(1)

  x = tf.keras.Input(shape=train_data.element_spec[0].shape[1:])
  out = tf.image.convert_image_dtype(x, tf.float32)
  out = fast_scnn(out, num_downsampling_layers=3, num_classes=2)
  model = tf.keras.Model(inputs=x, outputs=out)
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.Adam(learning_rate),
      # this does actually not ignore any of the 2 classes, but necessary because
      # standard MeanIoU does expect argmax output
      metrics=[IgnorantMeanIoU(num_classes=3, class_to_ignore=2)])
  history = model.fit(train_data,
                      epochs=epochs,
                      validation_data=val_data,
                      verbose=2)
  model.save(os.path.join(TMPDIR, 'model'))
  make_archive(os.path.join(TMPDIR, 'model.zip'), 'zip',
               os.path.join(TMPDIR, 'model'))
  _run.add_artifact(os.path.join(TMPDIR, 'model.zip'))
  hist = pd.DataFrame(history.history)
  for metric in hist.columns:
    _run.info[f'final_{metric}'] = hist[metric].iloc[-1]
  hist['epoch'] = history.epoch
  for _, row in hist.iterrows():
    for metric in hist.columns:
      if metric == 'epoch':
        continue
      _run.log_scalar(metric, row[metric], row['epoch'])


if __name__ == '__main__':
  ex.run_commandline()
