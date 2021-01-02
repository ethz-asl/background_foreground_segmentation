"""Nyu_depth_v2_labeled dataset."""
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
from skimage.transform import resize

# TODO(Nyu_depth_v2_labeled): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
The NYU-Depth V2 labeled data set is comprised of video sequences from a variety of indoor scenes as recorded by both the RGB and Depth cameras from the Microsoft Kinect.
It contains 1449 densely labeled pairs of aligned RGB and depth images.
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


def convert_mat(data):
  assert data.attrs['MATLAB_class'].decode() == 'char'
  string_array = np.array(data).ravel()
  string_array = ''.join([chr(x) for x in string_array])
  string_array = string_array.replace('\x00', '')
  return string_array


class NyuDepthV2Labeled(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Nyu_depth_v2_labeled dataset."""

  VERSION = tfds.core.Version('2.0.0')
  RELEASE_NOTES = {
      # '1.0.0': 'Initial release.',
      '2.0.0': 'different scenes for train_experiments/test',
  }

  # MANUAL_DOWNLOAD_INSTRUCTIONS = 1

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(Nyu_depth_v2_labeled): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(480, 640, 3), dtype=tf.uint8),
            'label': tfds.features.Tensor(shape=(480, 640,2), dtype=tf.int64),
        }),
        # features=tfds.features.FeaturesDict({
        #   'image': tfds.features.Image(shape=(48, 64, 3), dtype=tf.uint8),
        #   # 'depth': tfds.features.Tensor(shape=(480, 640), dtype=tf.float16),
        #   'label': tfds.features.Tensor(shape=(48, 64), dtype=tf.uint16),
        # }),
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
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
                'dataset_path': download_dir,
                # 'dataset_path': dl_manager.manual_dir,
                'scene_type': 'kitchen',
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                'dataset_path': download_dir,
                # 'dataset_path': dl_manager.manual_dir,
                'scene_type': 'bedroom',
            },
        ),
    ]


  def _generate_examples(self, dataset_path, scene_type):
    """Yields examples."""
    # TODO(Nyu_depth_v2_labeled): Yields (key, example) tuples from the dataset
    h5py = tfds.core.lazy_imports.h5py
    with h5py.File(dataset_path, 'r') as f:
      Images = f['images']
      # Depths = f['depths']
      Labels = f['labels']
      Images = np.array(f['images'], dtype=f['images'].dtype).T.squeeze()
      Depths = np.array(f['depths'], dtype=f['depths'].dtype).T.squeeze()
      Labels = np.array(f['labels'], dtype=f['labels'].dtype).T.squeeze()
      refs = f['#refs#']
      cell = []
      for ref in f['sceneTypes']:
        row = []
        for r in ref:
          entry = refs.get(r)
          row.append(convert_mat(entry))
        cell.append(row)
      for i in range(Images.shape[-1]):
        # Label_expand = np.expand_dims(Labels[:,:,i], axis=2)
        if cell[0][i] == scene_type:
          label = Labels[:, :, i]
          combine_label = np.logical_not(
              np.logical_or(label == 4,
                            (np.logical_or(label == 11, label == 21)))).astype(
                                int)
          yield str(i).zfill(4), {
              'image': Images[:, :, :, i],
              # 'depth': Depths[:, :, i],
              # 'label':mask}
              #'label': combine_label
              'label': np.stack([combine_label, (Depths[:, :, i]*100).astype(int)],axis = -1)
          }
        # yield str(i).zfill(4), {'image':Image_resize,
        #         'label':Label_resize}

# h5py = tfds.core.lazy_imports.h5py
# f=h5py.File('nyu_depth_v2_labeled.mat','r')
# refs = f['#refs#']
# cell = []
# for ref in f['sceneTypes']:
#   row = []
#   for r in ref:
#       entry = refs.get(r)
#       row.append(convert_mat(entry))
#   cell.append(row)


def count_scenes_num():
  h5py = tfds.core.lazy_imports.h5py
  f = h5py.File('nyu_depth_v2_labeled.mat', 'r')
  refs = f['#refs#']
  cell = []
  for ref in f['sceneTypes']:
    row = []
    for r in ref:
      entry = refs.get(r)
      row.append(convert_mat(entry))
    cell.append(row)
  scene_type_list = ['kitchen', 'office', 'bathroom', 'living_room', 'bedroom', 'bookstore', 'cafe', 'furniture_store', \
                'study_room', 'classroom', 'computer_lab', 'conference_room', 'dinette', 'excercise_room', 'foyer', \
                'home_office', 'home_storage', 'indoor_balcony', 'laundry_room', 'office_kitchen', 'playroom', \
                'printer_room', 'reception_room', 'study', 'basement', 'dining_room', 'student_lounge']
  #[225.,  78., 121., 221., 383.,  36.,   5.,  27.,  \
  #7.,  49.,   6., 5.,   4.,   3.,   4., \
  # 50.,   5.,   2.,   3.,  10.,  31.,   \
  # 3., 17.,  25.,   7., 117.,   5.]
  cnt = np.zeros(len(scene_type_list))
  for i, scene_type in enumerate(scene_type_list):
    for j in cell[0]:
      if j == scene_type:
        cnt[i] += 1
