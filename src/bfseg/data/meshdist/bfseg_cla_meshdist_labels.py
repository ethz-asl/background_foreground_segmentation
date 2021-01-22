import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

from bfseg.utils.utils import load_gdrive_file

_DESCRIPTION = """
This dataset contains data from the ARCHE as well as the CLA building.
It consists of three labels (0,1,2) where all classes that belong to the background (e.g. floor, wall, roof) are assigned the
'2' label. Background has assigned the '0' label and unknown the '1' label
"""


class BfsegCLAMeshdistLabels(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for BfsegCLAMeshdistLabels dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'Initial release.'}

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(builder=self,
                                 description=_DESCRIPTION,
                                 features=tfds.features.FeaturesDict({
                                     'image':
                                         tfds.features.Tensor(shape=(480, 640,
                                                                     3),
                                                              dtype=tf.float64),
                                     'label':
                                         tfds.features.Tensor(shape=(480, 640,
                                                                     1),
                                                              dtype=tf.uint8),
                                 }),
                                 supervised_keys=("image", "label"))

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    dataset_path = dl_manager.download(
        "https://drive.google.com/uc?export=download&id=1dJN2c4nP5Yh27Qjo49_3gCqhnLw-4Nz5"
    )
    return [
        tfds.core.SplitGenerator(
            name="fused",
            gen_kwargs={
                'dataset_path': dataset_path,
                'scene_type': "CLA_FUSED",
            },
        )
    ]

  def _generate_examples(self, dataset_path, scene_type):
    """Yields examples."""
    h5py = tfds.core.lazy_imports.h5py
    with h5py.File(dataset_path, 'r') as f:
      images = f[scene_type]['images']
      labels = f[scene_type]['labels']
      for i in range(images.shape[0]):
        yield str(i).zfill(4), {
            'image': images[i, ...],
            'label': labels[i, ...]
        }
