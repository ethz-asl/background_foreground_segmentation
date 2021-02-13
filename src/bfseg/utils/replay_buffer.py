import numpy as np
import tensorflow as tf

from bfseg.utils.images import augmentation


class ReplayBuffer:
  r"""Implements a replay buffer, which allows merging a given dataset with
  samples from another dataset. Merging can either keep the size of the main
  dataset - in which case samples from both datasets are randomly selected so as
  as to keep a desired ratio between the two datasets - or add the samples to
  replay on top of the main dataset - in which case a fraction of the dataset to
  to add needs to be specified. Optional online data augmentation is used.

  Args:
    main_ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset): Main
      dataset from which the samples are selected.
    replay_ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset): Replay
      dataset, from which selected samples are drawn and merged to the those
      from the main dataset (cf. docs).
    batch_size (int): Batch size to use for the merged dataset.
    ratio_main_ds_replay_ds (list of int): If not None, the two datasets are
      merged so that the resulting dataset has the same size as the main
      dataset, and samples are randomly drawn from the two dataset so that the
      ratio between the samples from the main dataset and those from the replay
      dataset matches this ratio. The ratio is expressed by a list of two
      integers: a ratio of '[3, 2]' means that 3 samples should be taken from
      the main dataset every 2 samples from the other dataset. NOTE: An error
      will be raised if the not enough samples from any of the two datasets can
      be taken. Exactly one between `ratio_main_ds_replay_ds` and
      `fraction_replay_ds_to_use` must be not None.
    fraction_replay_ds_to_use (float): If not None, the two datasets are merged
      so that the resulting dataset contains all samples from the main dataset,
      and a number of randomly selected samples from the replay dataset that
      matches this fraction. Exactly one between `ratio_main_ds_replay_ds` and
      `fraction_replay_ds_to_use` must be not None.
    perform_data_augmentation (bool): Whether or not online data augmentation
      should be performed on the samples from the output merged dataset.
  """

  def __init__(self,
               main_ds,
               replay_ds,
               batch_size,
               ratio_main_ds_replay_ds=None,
               fraction_replay_ds_to_use=None,
               perform_data_augmentation=True):
    assert ((ratio_main_ds_replay_ds is None) !=
            (fraction_replay_ds_to_use is None)), (
                "Exactly one between `ratio_main_ds_replay_ds` and "
                "`fraction_replay_ds_to_use` must be not None.")
    if (ratio_main_ds_replay_ds is not None):
      assert (isinstance(ratio_main_ds_replay_ds, list) and
              len(ratio_main_ds_replay_ds) == 2 and
              isinstance(ratio_main_ds_replay_ds[0], int) and
              isinstance(ratio_main_ds_replay_ds[1], int))
    if (fraction_replay_ds_to_use is not None):
      assert (isinstance(fraction_replay_ds_to_use, float) and
              0.0 <= fraction_replay_ds_to_use <= 1.0)
    assert (isinstance(perform_data_augmentation, bool))
    self._main_ds = main_ds
    self._replay_ds = replay_ds
    self._batch_size = batch_size
    self._ratio_main_ds_replay_ds = ratio_main_ds_replay_ds
    self._fraction_replay_ds_to_use = fraction_replay_ds_to_use
    self._perform_data_augmentation = perform_data_augmentation
    # Compute the number of samples in the two datasets.
    self._tot_num_samples_main = sum(
        sample[0].shape[0] for sample in self._main_ds)
    self._tot_num_samples_replay = sum(
        sample[0].shape[0] for sample in self._replay_ds)

  def _concat_samples(self, sample_1, sample_2):
    r"""Concatenates samples along the batch axis.
    """
    return (tf.concat((sample_1[0], sample_2[0]),
                      axis=0), tf.concat((sample_1[1], sample_2[1]), axis=0),
            tf.concat((sample_1[2], sample_2[2]), axis=0))

  def flow(self):
    r"""Performs one merge of the datasets, drawing a new set of samples to
    select.

    Modified from https://stackoverflow.com/a/58573644.

    Returns:
      merged_ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset): Merged
        dataset with the desired batch size.          
    """
    if (self._ratio_main_ds_replay_ds is not None):
      ratio_main, ratio_replay = self._ratio_main_ds_replay_ds
      # The main and the replay datasets are 'zipped' to form the merged
      # dataset. The number of 'batches' to take from the zipped dataset is
      # determined by the total number of samples to have (equal to original
      # number of samples in the main dataset) and by the ratio that should be
      # kept.
      zipped_ds_batch_size = ratio_main + ratio_replay
      zipped_ds_num_batches = int(
          np.ceil(self._tot_num_samples_main / zipped_ds_batch_size))
      if (self._tot_num_samples_replay < zipped_ds_num_batches * ratio_replay):
        raise ValueError(
            f"Asked to replay {zipped_ds_num_batches * ratio_replay} samples "
            "from the replay dataset, but this only has "
            f"{self._tot_num_samples_replay} samples.")

      total_num_samples = zipped_ds_batch_size * zipped_ds_num_batches

      # Take the required number of elements (selected randomly) from each
      # dataset, so as to keep the desired ratio.
      subset_main_ds = self._main_ds.unbatch().shuffle(
          self._tot_num_samples_main).batch(ratio_main).take(
              zipped_ds_num_batches)
      subset_replay_ds = self._replay_ds.unbatch().shuffle(
          self._tot_num_samples_replay).batch(ratio_replay).take(
              zipped_ds_num_batches)
      # Zip the two datasets to form the merged dataset.
      merged_ds = tf.data.Dataset.zip(
          (subset_main_ds,
           subset_replay_ds)).map(self._concat_samples).unbatch()
    else:
      # Keep the main dataset as it is.
      subset_main_ds = self._main_ds.unbatch()
      num_replay_samples_to_keep = int(
          np.ceil(self._fraction_replay_ds_to_use *
                  self._tot_num_samples_replay))
      subset_replay_ds = self._replay_ds.unbatch().shuffle(
          self._tot_num_samples_replay).take(num_replay_samples_to_keep)
      merged_ds = subset_main_ds.concatenate(subset_replay_ds)

      total_num_samples = (self._tot_num_samples_main +
                           num_replay_samples_to_keep)

    # Optionally perform data augmentation.
    if (self._perform_data_augmentation):
      merged_ds = merged_ds.map(augmentation)

    # Shuffle, batch and return the dataset.
    merged_ds = merged_ds.shuffle(total_num_samples)
    merged_ds = merged_ds.batch(self._batch_size).prefetch(
        tf.data.experimental.AUTOTUNE)

    return merged_ds
