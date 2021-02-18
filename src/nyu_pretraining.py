from sacred import Experiment
import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
import os
from shutil import make_archive

import bfseg.data.nyu.Nyu_depth_v2_labeled
from bfseg.utils.metrics import IgnorantMeanIoU, IgnorantAccuracyMetric
from bfseg.utils.models import create_model
from bfseg.utils.utils import crop_map
from bfseg.utils.losses import IgnorantCrossEntropyLoss, BalancedIgnorantCrossEntropyLoss
from bfseg.utils.images import augmentation
from bfseg.settings import TMPDIR
from bfseg.sacred_utils import get_observer

ex = Experiment()
ex.observers.append(get_observer())


@ex.main
def pretrain_nyu(_run,
                 batchsize=10,
                 epochs=100,
                 learning_rate=1e-4,
                 stopping_patience=50,
                 data_augmentation=True,
                 balanced_loss=True,
                 normalization_type='group',
                 test=False):
  train_data = tfds.load(
      'NyuDepthV2Labeled', split='full[:90%]',
      as_supervised=True).map(crop_map).shuffle(1000).batch(batchsize).cache()
  if data_augmentation:
    train_data = train_data.map(augmentation)
  val_data = tfds.load(
      'NyuDepthV2Labeled', split='full[90%:]',
      as_supervised=True).map(crop_map).batch(batchsize).cache()
  if test:
    # for testing
    train_data = train_data.take(1)
    val_data = val_data.take(1)

  image_h, image_w = train_data.element_spec[0].shape[1:3]
  _, model = create_model(model_name="fast_scnn",
                          image_h=image_h,
                          image_w=image_w,
                          freeze_encoder=False,
                          freeze_whole_model=False,
                          normalization_type="batch",
                          num_downsampling_layers=2)
  if balanced_loss:
    loss = BalancedIgnorantCrossEntropyLoss(class_to_ignore=2,
                                            num_classes=3,
                                            from_logits=True)
  else:
    loss = IgnorantCrossEntropyLoss(class_to_ignore=2,
                                    num_classes=3,
                                    from_logits=True)
  model.compile(
      loss=loss,
      optimizer=tf.keras.optimizers.Adam(learning_rate),
      # this does actually not ignore any of the 2 classes, but necessary because
      # standard MeanIoU does expect argmax output
      metrics=[
          IgnorantMeanIoU(num_classes=3, class_to_ignore=2),
          IgnorantAccuracyMetric(num_classes=3, class_to_ignore=2)
      ])
  history = model.fit(
      train_data,
      epochs=epochs,
      validation_data=val_data,
      verbose=2,
      callbacks=[
          tf.keras.callbacks.ReduceLROnPlateau(),
          tf.keras.callbacks.EarlyStopping(patience=stopping_patience)
      ])
  modelpath = os.path.join(TMPDIR, 'model')
  model.save(modelpath)
  make_archive(modelpath, 'zip', modelpath)
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
  return float(hist['val_ignorant_mean_io_u'].iloc[-1])


if __name__ == '__main__':
  ex.run_commandline()
