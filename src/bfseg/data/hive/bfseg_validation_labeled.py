import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

from bfseg.utils.utils import load_gdrive_file

_DESCRIPTION = """
This dataset contains data from the ARCHE as well as the CLA building.
It consists of two labels (0,1) where all classes that belong to the background (e.g. floor, wall, roof) are assigned the
'1' label.
"""


class BfsegValidationLabeled(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for BfsegValidationLabeled dataset."""

  VERSION = tfds.core.Version('1.0.1')
  RELEASE_NOTES = {'1.0.0': 'Initial release.', '1.0.1': 'Added Names'}

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
                                     'name':
                                         tf.string
                                 }),
                                 supervised_keys=("image", "label"))

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    dataset_path = dl_manager.download(
        "https://drive.google.com/uc?export=download&id=1qpDnnxSCqNR3LrOv6kkVrdeU8yotSkXB"
    )

    return [
        tfds.core.SplitGenerator(
            name="CLA",
            gen_kwargs={
                'dataset_path': dataset_path,
                'scene_type': "CLA",
            },
        ),
        tfds.core.SplitGenerator(
            name="ARCHE",
            gen_kwargs={
                'dataset_path': dataset_path,
                'scene_type': "ARCHE",
            },
        ),
    ]

  def _generate_examples(self, dataset_path, scene_type):
    """Yields examples."""
    h5py = tfds.core.lazy_imports.h5py
    with h5py.File(dataset_path, 'r') as f:
      images = f[scene_type]['images']
      labels = f[scene_type]['labels']
      metadata = f['metadata'][scene_type]
      for i in range(images.shape[0]):
        cam = [key for key in metadata[str(i)].keys()][0]
        timestamp = metadata[str(i)][cam][0][0]
        yield str(i).zfill(4), {
            'image': images[i, ...],
            # Labels from the hive set are
            # (0: Other,  87: Floor, 118: Wall, 134: Roof)
            # remap them to (0: Foreground, 1: Background)
            'label': (labels[i, ...] != 0).astype(np.uint8),
            'name': cam + "_" + str(timestamp)
        }
