from os import path, environ, listdir, mkdir
from zipfile import ZipFile
import json
import tensorflow as tf
import numpy as np
import cv2
from bfseg.settings import TMPDIR
from shutil import rmtree


def load_fsdata(base_path, dataset_info=None, modalities=None):
  if not base_path.startswith('/') or not path.exists(base_path):
    message = 'ERROR: Path to CITYSCAPES dataset does not exist.'
    print(message)
    raise IOError(1, message, base_path)

  if base_path.endswith('.zip'):
    print('INFO Loading into machine ... ')
    tmppath = path.join(TMPDIR, 'fsdata_tmp')
    rmtree(tmppath)
    mkdir(tmppath)
    with ZipFile(base_path, 'r') as arch:
      arch.extractall(path=tmppath)
    base_path = tmppath
    print('DONE')

  all_files = listdir(base_path)
  if dataset_info is None and 'dataset_info.json' not in all_files:
    message = 'ERROR: folder does not contain dataset_info.json'
    print(message)
    raise IOError(1, message, base_path)
  if dataset_info is None:
    # load the info file and check whether it is consistent with info from other
    # sets
    with open(path.join(base_path, 'dataset_info.json'), 'r') as f:
      dataset_info = json.load(f)

  if 'dataset_info.json' in all_files:
    all_files.remove('dataset_info.json')
  # now group filenames by their prefixes
  grouped_by_idx = {}
  for filename in sorted(all_files):
    prefix = '_'.join(filename.split('_')[:-1])
    grouped_by_idx.setdefault(prefix, []).append(path.join(base_path, filename))

  data_shape_description = dataset_info['output_shapes']
  data_shape_description['filename'] = []
  name_to_tf_type = {
      'int32': tf.int32,
      'float32': tf.float32,
      'string': tf.string
  }
  data_type_description = {
      m: name_to_tf_type[name]
      for m, name in dataset_info['output_types'].items()
  }
  data_type_description['filename'] = tf.string

  def _get_data():
    nonlocal grouped_by_idx
    for prefix, item in grouped_by_idx.items():
      blob = {'filename': prefix}
      for filepath in item:
        components = filepath.split('_')
        modality, filetype = components[-1].split('.')
        if modalities is not None and modality not in modalities:
          # skip this modality
          continue
        if filetype == 'npz':
          blob[modality] = np.load(filepath)[modality]
        elif filetype == 'npy':
          blob[modality] = np.load(filepath)
        elif modality == 'rgb':
          img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
          if img.shape[2] == 4:
            rgb = img[:, :, :3][..., ::-1]
            alpha = img[:, :, 3]
            blob['rgb'] = np.concatenate((rgb, np.expand_dims(alpha, -1)),
                                         axis=-1)
          else:
            blob['rgb'] = img[..., ::-1]
        else:
          data = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
          if modality in ['labels', 'mask']:
            # opencv translation as it ignores negative values
            data = data.astype(dataset_info['output_types'][modality])
            data[data == 255] = -1
            blob[modality] = data
      yield blob

  return tf.data.Dataset.from_generator(_get_data, data_type_description,
                                        data_shape_description)


def dump_dataset(dataset,
                 directory: str,
                 only_modalities=None,
                 num_classes=None,
                 all_as_np=False,
                 use_name=False):
  """Writes dataset to file structure."""
  if not path.exists(directory):
    message = 'ERROR: Path for dataset dump does not exist.'
    print(message)
    raise IOError(1, message, directory)

  for idx, blob in enumerate(dataset.as_numpy_iterator()):
    for m in blob:
      if only_modalities is not None and m not in only_modalities:
        continue
      if 'name' in blob:
        name = copy(blob['name'])
        name = name.decode() if hasattr(name, 'decode') else name
        filename = '{:04d}_{}_{}'.format(idx, name, m)
      else:
        filename = '{:04d}_{}'.format(idx, m)

      if m in ['rgb', 'visual'] and not all_as_np:
        if blob[m].shape[2] == 4:
          # convert to BGRA
          bgr = blob[m].astype('uint8')[..., :3][..., ::-1]
          a = blob[m].astype('uint8')[..., 3:]
          data = np.concatenate((bgr, a), axis=-1)
        else:
          # convert to BGR
          data = blob[m].astype('uint8')[..., ::-1]
        cv2.imwrite(path.join(directory, filename + '.png'), data)
      elif (blob[m].ndim > 2 and blob[m].shape[2] > 1) \
              or all_as_np:
        # save as numpy array
        np.savez_compressed(path.join(directory, filename + '.npz'),
                            **{m: blob[m]})
      else:
        data = blob[m]
        # translate -1 values into something open-cv does not ignore
        data[data == -1] = 255
        data = data.astype('uint8')
        cv2.imwrite(path.join(directory, filename + '.png'), data)

  # write a description of the data
  info = {
      'output_shapes': {
          m: list(spec.shape)
          for m, spec in dataset.element_spec.items()
          if not (only_modalities is not None and m not in only_modalities)
      },
      'output_types': {
          m: spec.dtype.name
          for m, spec in dataset.element_spec.items()
          if not (only_modalities is not None and m not in only_modalities)
      }
  }
  if num_classes is not None:
    info['num_classes'] = num_classes
  with open(path.join(directory, 'dataset_info.json'), 'w') as f:
    json.dump(info, f)
