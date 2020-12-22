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

_CITATION = """
Should we do this?
"""


class BfsegValidationLabeled(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Nyu_depth_v2_labeled dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'Initial release.'}

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image':
                tfds.features.Tensor(shape=(480, 640, 3), dtype=tf.float64),
            'label':
                tfds.features.Tensor(shape=(480, 640, 1), dtype=tf.uint8),
        }),
        supervised_keys=("image", "label"),
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    download_dir = os.path.expanduser(
        "~/tensorflow_datasets/hive_labels_validation/")
    file_dir = download_dir + "data.h5"

    if not os.path.exists(file_dir):
      os.makedirs(download_dir, exist_ok=True)
      print("Data file does not exist, going to download it from google drive")
      saved_file = load_gdrive_file('1qpDnnxSCqNR3LrOv6kkVrdeU8yotSkXB',
                                    output_folder=download_dir)
      _, filename = os.path.split(saved_file)
      os.rename(saved_file, saved_file.replace(filename, "data.h5"))

    return [
        tfds.core.SplitGenerator(
            name="CLA",
            gen_kwargs={
                'dataset_path': download_dir + "data.h5",
                'scene_type': "CLA",
            },
        ),
        tfds.core.SplitGenerator(
            name="ARCHE",
            gen_kwargs={
                'dataset_path': download_dir + "data.h5",
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
      for i in range(images.shape[0]):
        yield str(i).zfill(4), {
            'image': images[i, ...],
            # Labels from the hive set are
            # (0: Other,  87: Floor, 118: Wall, 134: Roof)
            # remap them to (0: Foreground, 1: Background)
            'label': (labels[i, ...] != 0).astype(np.uint8)
        }
