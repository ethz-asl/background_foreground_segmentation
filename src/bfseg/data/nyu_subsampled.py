import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import json
import cv2

_DESCRIPTION = """
A perfectly random subsampled version of 144 NYU images and labels.
"""


class NyuSubsampled(tfds.core.GeneratorBasedBuilder):

  VERSION = tfds.core.Version('0.1.0')
  RELEASE_NOTES = {'0.1.0': 'Initial testing.'}

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(480, 640, 3), dtype=tf.uint8),
            'label': tfds.features.Image(shape=(480, 640, 1), dtype=tf.uint8),
        }),
        supervised_keys=("image", "label"))

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    dataset_path = dl_manager.download_and_extract(
        'https://drive.google.com/uc?export=download&id=1mzC_hF_JbfXhLmo3cnRipgoGhmPizvxV'
    )
    return [
        tfds.core.SplitGenerator(
            name='full',
            gen_kwargs={
                'dataset_path': dataset_path,
            },
        )
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
      blob = {}
      for filepath in item:
        components = filepath.split('_')
        modality, filetype = components[-1].split('.')
        if filetype == 'npz':
          blob[modality] = np.load(filepath)[modality]
        elif filetype == 'npy':
          blob[modality] = np.load(filepath)
        elif modality in ['rgb', 'image']:
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
          if modality in ['labels', 'label', 'mask']:
            data = data.astype(dataset_info['output_types'][modality])
            # image must be 3 dimensional in tensorflow
            blob['label'] = np.expand_dims(data.astype('uint8'), axis=-1)
      yield prefix, blob
