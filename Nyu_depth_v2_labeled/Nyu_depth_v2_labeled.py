"""Nyu_depth_v2_labeled dataset."""
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

# TODO(Nyu_depth_v2_labeled): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(Nyu_depth_v2_labeled): BibTeX citation
_CITATION = """
@inproceedings{Silberman:ECCV12,
  author    = {Nathan Silberman, Derek Hoiem, Pushmeet Kohli and Rob Fergus},
  title     = {Indoor Segmentation and Support Inference from RGBD Images},
  booktitle = {ECCV},
  year      = {2012}
}
"""

_URL = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'

class NyuDepthV2Labeled(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Nyu_depth_v2_labeled dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(Nyu_depth_v2_labeled): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
          'image': tfds.features.Image(shape=(480, 640, 3), dtype=tf.uint8),
          # 'depth': tfds.features.Tensor(shape=(480, 640), dtype=tf.float16),
          'label': tfds.features.Tensor(shape=(480, 640), dtype=tf.uint16), 
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=("image", "label"),  # e.g. ('image', 'label')
        homepage='https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(Nyu_depth_v2_labeled): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    download_dir = dl_manager.download(_URL)
    return [
        tfds.core.SplitGenerator(
            'train',
            # These kwargs will be passed to _generate_examples
            gen_kwargs={'dataset_path': download_dir},
        ),
    ]

  def _generate_examples(self, dataset_path):
    """Yields examples."""
    # TODO(Nyu_depth_v2_labeled): Yields (key, example) tuples from the dataset
    h5py = tfds.core.lazy_imports.h5py
    with h5py.File(dataset_path,'r') as f:
      Images = f['images']
      # Depths = f['depths']
      Labels = f['labels']
      Images=np.array(f['images'],dtype=f['images'].dtype).T.squeeze()
      # Depths=np.array(f['depths'],dtype=f['images'].dtype).T.squeeze()
      Labels=np.array(f['labels'],dtype=f['labels'].dtype).T.squeeze()
      for i in range(Images.shape[-1]):
          yield str(i).zfill(4), {'image':Images[:,:,:,i],
                 # 'depth':Depths[:,:,i],
                 'label':Labels[:,:,i]}
