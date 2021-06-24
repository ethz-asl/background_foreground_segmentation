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


class MeshdistPseudolabelsDense(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for BfsegCLAMeshdistLabels dataset."""

  VERSION = tfds.core.Version('0.2.0')
  RELEASE_NOTES = {
      '0.2.0': 'added office scenes',
      '0.1.3': 'completed alphasense garage data',
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
        'office6_sparse50_allcams': 
            'https://drive.google.com/uc?export=download&id=1dGiML_JyvJF1qkEF16p8Xn7KQDl9BE6s',
        'office6_paper':
            'https://drive.google.com/uc?export=download&id=1h0SSB1tRY76t4WvAKpH6AnguAsxSHWlv',
        'office12_dense20_dyn_cam2':
            'https://drive.google.com/uc?export=download&id=1VbD8N_T9HqbEzaBhy53YlUYEVd-gPq_C',
        'office12_sparse50_dyn_cam2':
            'https://drive.google.com/uc?export=download&id=1NF2aJ_-jjjuaW0Wox-wwVd-rcBwwSlQB',
        'office12_sparse50_dyn_allcams':
            'https://drive.google.com/uc?export=download&id=11kedzHhymovz6QDdEw0K9lbkXf1RO2TH',
        'office3_dense20_dyn_cam2':
            'https://drive.google.com/uc?export=download&id=1yjexvTgQDfM_5V6-XAVH-1Y9Sj0koiTp',
        'office3_sparse50_dyn_cam2':
            'https://drive.google.com/uc?export=download&id=1aaCaFXkGUd503aPql50YOUHnq6CJDdTm',
        'office3_sparse50_dyn_allcams':
            'https://drive.google.com/uc?export=download&id=1b--TyIidfTZVoL4aYrBUFXe9_S7rqF2N',
        'office3_combined2050_dyn_cam2':
            'https://drive.google.com/uc?export=download&id=1IpNZPdwvBO7pnZJVEtK-_hq8W78F8t82'
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