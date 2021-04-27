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
    replay_ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset or list):
      Single replay dataset or list of replay datasets from which selected
      samples are drawn and merged to the those from the main dataset (cf.
      docs).
    batch_size (int): Batch size to use for the merged dataset.
    ratio_main_ds_replay_ds (list of int): If not None, the datasets are merged
      so that the resulting dataset has (approximately) the same size as the
      main dataset, and samples are randomly drawn from the both the main
      dataset and the replay dataset(s) in such a way that the ratio between the
      samples from the main dataset and those from the replay dataset(s) matches
      this ratio. The ratio is expressed by a list of as `num_replay_datasets`+1
      integers, where `num_replay_datasets` is the number of replay datasets:
      for instance, a ratio of '[3, 2, 1]' means that 3 samples should be taken
      from the main dataset every 2 samples from the first replay dataset and
      every sample from the second replay dataset. NOTE: An error will be raised
      if the not enough samples from any of the datasets can be taken. Exactly
      one between `ratio_main_ds_replay_ds` and `fraction_replay_ds_to_use` must
      be not None.
    fraction_replay_ds_to_use (float or list of float): If not None, the main
      dataset and the replay dataset(s) are merged so that the resulting dataset
      contains all samples from the main dataset, and a number of randomly
      selected samples from each the replay dataset that matches the fraction
      associated to each replay dataset. If a single value is given, the same
      fraction will be used for all the replay datasets. Exactly one between
      `ratio_main_ds_replay_ds` and `fraction_replay_ds_to_use` must be not
      None.
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
    assert (
        replay_ds is not None
    ), "Replay dataset must be specified in order to build the replay buffer."
    assert ((ratio_main_ds_replay_ds is None) !=
            (fraction_replay_ds_to_use is None)), (
                "Exactly one between `ratio_main_ds_replay_ds` and "
                "`fraction_replay_ds_to_use` must be not None.")
    if (isinstance(replay_ds, list)):
      total_num_datasets = len(replay_ds) + 1
    else:
      total_num_datasets = 2
    if (ratio_main_ds_replay_ds is not None):
      assert (isinstance(ratio_main_ds_replay_ds, list) and
              len(ratio_main_ds_replay_ds) == total_num_datasets)
      for ratio in ratio_main_ds_replay_ds:
        assert (isinstance(ratio, int))
    if (fraction_replay_ds_to_use is not None):
      if (isinstance(fraction_replay_ds_to_use, list)):
        assert (len(fraction_replay_ds_to_use) == total_num_datasets - 1)
      else:
        fraction_replay_ds_to_use = [
            fraction_replay_ds_to_use for _ in range(total_num_datasets - 1)
        ]
        if (total_num_datasets > 2):
          print(
              "Will use the same replay fraction for all the replay datasets.")

      for replay_fraction in fraction_replay_ds_to_use:
        assert (isinstance(replay_fraction, float) and
                (0.0 <= replay_fraction <= 1.0))
    assert (isinstance(perform_data_augmentation, bool))
    self._main_ds = main_ds
    if (isinstance(replay_ds, list)):
      self._replay_datasets = replay_ds
    else:
      self._replay_datasets = [replay_ds]
    self._batch_size = batch_size
    self._ratio_main_ds_replay_ds = ratio_main_ds_replay_ds
    self._fraction_replay_ds_to_use = fraction_replay_ds_to_use
    self._perform_data_augmentation = perform_data_augmentation
    # Compute the number of samples in the two datasets.
    self._tot_num_samples_main = sum(
        sample[0].shape[0] for sample in self._main_ds)
    self._tot_num_samples_replay = [
        sum(sample[0].shape[0]
            for sample in replay_dataset)
        for replay_dataset in self._replay_datasets
    ]

  def _concat_samples(self, *samples):
    r"""Concatenates samples along the batch axis.
    """
    return (tf.concat(tuple(sample[0] for sample in samples), axis=0),
            tf.concat(tuple(sample[1] for sample in samples), axis=0),
            tf.concat(tuple(sample[2] for sample in samples), axis=0))

  def flow(self):
    r"""Performs one merge of the datasets, drawing a new set of samples to
    select.
    Modified from https://stackoverflow.com/a/58573644.
    Returns:
      merged_ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset): Merged
        dataset with the desired batch size.
    """
    if (self._ratio_main_ds_replay_ds is not None):
      ratio_main = self._ratio_main_ds_replay_ds[0]
      ratios_replay = [ratio for ratio in self._ratio_main_ds_replay_ds[1:]]
      # The main and the replay datasets are 'zipped' to form the merged
      # dataset. The number of 'batches' to take from the zipped dataset is
      # determined by the total number of samples to have (equal to original
      # number of samples in the main dataset) and by the ratio that should be
      # kept.
      zipped_ds_batch_size = ratio_main + sum(ratios_replay)
      zipped_ds_num_batches = int(
          np.ceil(self._tot_num_samples_main / zipped_ds_batch_size))
      # Check on the number of samples.
      for replay_ds_idx, ratio_replay in enumerate(ratios_replay):
        num_samples_curr_replay_ds = self._tot_num_samples_replay[replay_ds_idx]
        if (num_samples_curr_replay_ds < zipped_ds_num_batches * ratio_replay):
          raise ValueError(
              f"Asked to replay {zipped_ds_num_batches * ratio_replay} samples "
              f"from the replay dataset with index {replay_ds_idx}, but this "
              f"has only {num_samples_curr_replay_ds} samples.")

      total_num_samples = zipped_ds_batch_size * zipped_ds_num_batches

      # Take the required number of elements (selected randomly) from each
      # dataset, so as to keep the desired ratio.
      subset_main_ds = self._main_ds.unbatch().shuffle(
          self._tot_num_samples_main).batch(ratio_main).take(
              zipped_ds_num_batches)
      subset_replay_ds = []
      for replay_ds_idx, ratio_replay in enumerate(ratios_replay):
        subset_replay_ds.append(
            self._replay_datasets[replay_ds_idx].unbatch().shuffle(
                self._tot_num_samples_replay[replay_ds_idx]).batch(
                    ratio_replay).take(zipped_ds_num_batches))
      # Zip the datasets to form the merged dataset.
      merged_ds = tf.data.Dataset.zip(
          (subset_main_ds,
           *subset_replay_ds)).map(self._concat_samples).unbatch()
    else:
      # Keep the main dataset as it is.
      subset_main_ds = self._main_ds.unbatch()
      num_replay_samples_to_keep = [
          int(
              np.ceil(replay_fraction_curr_replay_ds *
                      tot_num_samples_curr_replay_ds))
          for (replay_fraction_curr_replay_ds, tot_num_samples_curr_replay_ds)
          in zip(self._fraction_replay_ds_to_use, self._tot_num_samples_replay)
      ]
      merged_ds = subset_main_ds
      # Concatenate the required fraction of each replay dataset to the main
      # dataset.
      for (curr_replay_ds, tot_num_samples_curr_replay_ds,
           num_samples_to_keep_curr_replay_ds) in zip(
               self._replay_datasets, self._tot_num_samples_replay,
               num_replay_samples_to_keep):
        subset_replay_ds = curr_replay_ds.unbatch().shuffle(
            tot_num_samples_curr_replay_ds).take(
                num_samples_to_keep_curr_replay_ds)
        merged_ds = merged_ds.concatenate(subset_replay_ds)

      total_num_samples = (self._tot_num_samples_main +
                           sum(num_replay_samples_to_keep))

    # Batch, cache, and shuffle the merged dataset.
    merged_ds = merged_ds.cache().shuffle(total_num_samples).batch(
        self._batch_size)

    # Optionally perform data augmentation.
    if (self._perform_data_augmentation):
      merged_ds = merged_ds.map(augmentation)

    merged_ds = merged_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return merged_ds