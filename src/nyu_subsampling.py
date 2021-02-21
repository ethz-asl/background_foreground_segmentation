from sacred import Experiment
import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
import os
from shutil import make_archive

import bfseg.data.nyu.Nyu_depth_v2_labeled
from bfseg.data.fsdata import dump_dataset
from bfseg.settings import TMPDIR

ex = Experiment()


@ex.main
def subsample():
  nyu = tfds.load(
      'NyuDepthV2Labeled', split='full',
      as_supervised=True)
  num_samples = int(0.1 * len(nyu))
  print('Will sample {} images out of {} NYU images'.format(num_samples, len(nyu)), flush=True)
  samples = nyu.shuffle(len(nyu)).take(num_samples)
  tf.data.experimental.save(samples, '/cluster/work/riner/users/blumh/nyu_subsampled', compression='gzip')

  def make_dict(*blob):
      return {'rgb': blob[0], 'label': blob[1]}
  fsdata_path = '/cluster/work/riner/users/blumh/nyu_subsampled_fsdata'
  dump_dataset(samples.map(make_dict), fsdata_path)
  make_archive(fsdata_path, 'zip', fsdata_path)


if __name__ == '__main__':
  ex.run_commandline()
