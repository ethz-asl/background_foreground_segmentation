from sacred import Experiment
import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
import os
from shutil import make_archive
from zipfile import ZipFile

import bfseg.data.nyu.Nyu_depth_v2_labeled
from bfseg.data.fsdata import load_fsdata
from bfseg.utils.metrics import IgnorantMeanIoU, IgnorantAccuracyMetric
from bfseg.utils.losses import IgnorantCrossEntropyLoss, BalancedIgnorantCrossEntropyLoss
from bfseg.models.fast_scnn import fast_scnn
from bfseg.utils.utils import crop_map, load_gdrive_file
from bfseg.utils.images import resize_with_crop, augmentation
from bfseg.settings import TMPDIR
from bfseg.sacred_utils import get_observer
from bfseg.utils.models import create_model

ex = Experiment()
ex.observers.append(get_observer())


@ex.main
def finetuning(_run,
               datapath,
               pretrained_id,
               batchsize=10,
               epochs=100,
               learning_rate=1e-4,
               remove_cam2=False,
               stopping_patience=50,
               data_augmentation=True,
               balanced_loss=True,
               normalization_type='group',
               test=False):
  dataset_info = {
      'output_shapes': {
          'labels': [None, None],
          'rgb': [None, None, 3]
      },
      'output_types': {
          'labels': 'int32',
          'rgb': 'int32',
      }
  }
  train_data = load_fsdata(datapath, dataset_info=dataset_info)

  def cam2_remover(blob):
    return not tf.strings.regex_full_match(blob['filename'], '.*cam2$')

  if remove_cam2:
    train_data = train_data.filter(cam2_remover)

  # preprocesing and flatten the training data into touples of 'rgb', 'labels'
  def preprocessing(blob):
    rgb = blob['rgb']
    labels = blob['labels']
    # resizing
    rgb = tf.image.resize(rgb, (6 * 32, 11 * 32),
                          method=tf.image.ResizeMethod.BILINEAR)
    # label needs additional dimension for resizing
    labels = tf.image.resize(labels[..., tf.newaxis], (6 * 32, 11 * 32),
                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return rgb, labels

  train_data = train_data.map(preprocessing).cache()
  val_data = train_data.take(50).batch(batchsize)
  train_data = train_data.skip(50).shuffle(1000)
  if data_augmentation:
    train_data = train_data.map(augmentation)
  train_data = train_data.batch(batchsize)

  # resizing of validation datasets to same size
  def resizing(image, label):
    label = tf.reshape(label, (480, 640, 1))
    image = resize_with_crop(image, (6 * 32, 11 * 32))
    label = resize_with_crop(label, (6 * 32, 11 * 32),
                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image, label

  valnyu_data = tfds.load(
      'NyuDepthV2Labeled', split='full[90%:]',
      as_supervised=True).map(resizing).batch(batchsize).cache()
  valcla_data = tfds.load(
      "BfsegValidationLabeled", split="CLA",
      as_supervised=True).map(resizing).batch(batchsize).cache()

  # load the pretrained model
  ZipFile(load_gdrive_file(pretrained_id, ending='zip')).extractall(
      os.path.join(TMPDIR, 'pretrained_model'))
  pretrained = tf.keras.models.load_model(
      os.path.join(TMPDIR, 'pretrained_model'),
      custom_objects={"IgnorantMeanIoU": IgnorantMeanIoU},
      compile=False)

  _, model = create_model(model_name='fast_scnn',
                          image_h=train_data.element_spec[0].shape[1],
                          image_w=train_data.element_spec[0].shape[2],
                          freeze_encoder=False,
                          freeze_whole_model=False,
                          normalization_type=normalization_type,
                          num_downsampling_layers=2)

  pretrained.save_weights(os.path.join(TMPDIR, 'pretrained_weights'))
  model.load_weights(os.path.join(TMPDIR, 'pretrained_weights'))
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
  for metric, val in model.evaluate(train_data, return_dict=True).items():
    _run.log_scalar(metric, val, 0)
  for metric, val in model.evaluate(val_data, return_dict=True).items():
    _run.log_scalar("val_{}".format(metric), val, 0)
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
  _run.info['cla_hive'] = model.evaluate(valcla_data, return_dict=True)
  _run.info['nyu'] = model.evaluate(valnyu_data, return_dict=True)
  hist = pd.DataFrame(history.history)
  for metric in hist.columns:
    _run.info[f'final_{metric}'] = hist[metric].iloc[-1]
  hist['epoch'] = history.epoch
  for _, row in hist.iterrows():
    for metric in hist.columns:
      if metric == 'epoch':
        continue
      _run.log_scalar(metric, row[metric], row['epoch'])
  return float(hist['val_ignorant_mean_io_u_1'].iloc[-1])


if __name__ == '__main__':
  ex.run_commandline()
