import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import json
import cv2

_DESCRIPTION = """
Pseudolabels generated from all bagfiles used for the RSS experiments.
Labels:
  0 - foreground
  1 - background
  2 - unsure (ignore in training)
"""


class MeshdistPseudolabels(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for bagfile datasets (e.g., Rumlang)."""

  VERSION = tfds.core.Version('0.2.0')
  RELEASE_NOTES = {
      '0.2.0': 'added office scenes',
      '0.1.3': 'completed anonymous_sensor1 garage data',
      '0.1.2': 'removed cam2 from garage1',
      '0.1.1': 'more data',
      '0.1.0': 'Initial testing.'
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(None, None, 3), dtype=tf.uint8),
            'label': tfds.features.Image(shape=(None, None, 1), dtype=tf.uint8),
            'filename': tf.string,
        }),
        supervised_keys=("image", "label"))

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    dataset_paths = dl_manager.download_and_extract({
        'rumlang3':
            'https://drive.google.com/uc?export=download&id=1f-AI4eFwJPh_ZK4-4HrbAdVts6cZ-hWI',
        'rumlang2':
            'https://drive.google.com/uc?export=download&id=1GvXpFIbnsh2VbGjqMoL9tq8Skl3KI_6G',
        'garage1':
            'https://drive.google.com/uc?export=download&id=1G83Mcjp6SNdUcg7HIG74fh3xqjW8Fhbe',
        'garage2':
            'https://drive.google.com/uc?export=download&id=1KFbc7IP1hp9KNZCcV_JkLY3snyU88dTZ',
        'garage3':
            'https://drive.google.com/uc?export=download&id=1wWGBxsGo8tK98JKKBsUA9JB18OlKM_QE',
    })
    return [
        tfds.core.SplitGenerator(
            name=name_of_run,
            gen_kwargs={
                'dataset_path': dataset_path,
            },
        ) for name_of_run, dataset_path in dataset_paths.items()
    ]

  def _generate_examples(self, dataset_path):
    """Yields examples, similar to load_fsdata."""
    with open(os.path.join(dataset_path, 'dataset_info.json'), 'r') as f:
      dataset_info = json.load(f)
    all_files = tf.io.gfile.listdir(dataset_path)
    all_files.remove('dataset_info.json')
    # now group filenames by their prefixes
    grouped_by_idx = {}
    for filename in sorted(all_files):
      prefix = '_'.join(filename.split('_')[:-1])
      grouped_by_idx.setdefault(prefix,
                                []).append(os.path.join(dataset_path, filename))
    # and extract per prefix a set of images
    for prefix, item in grouped_by_idx.items():
      blob = {'filename': prefix}
      for filepath in item:
        components = filepath.split('_')
        modality, filetype = components[-1].split('.')
        if filetype == 'npz':
          blob[modality] = np.load(filepath)[modality]
        elif filetype == 'npy':
          blob[modality] = np.load(filepath)
        elif modality == 'rgb':
          img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
          if img.shape[2] == 4:
            rgb = img[:, :, :3][..., ::-1]
            alpha = img[:, :, 3]
            blob['image'] = np.concatenate((rgb, np.expand_dims(alpha, -1)),
                                           axis=-1)
          else:
            blob['image'] = img[..., ::-1]
        else:
          data = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
          if modality in ['labels', 'mask']:
            data = data.astype(dataset_info['output_types'][modality])
            # image must be 3 dimensional in tensorflow
            blob['label'] = np.expand_dims(data.astype('uint8'), axis=-1)
      yield prefix, blob
