import tensorflow as tf

from bfseg.data.fsdata import load_fsdata
from bfseg.utils.datasets import (load_data,
                                  preprocess_bagfile_different_dataloader)


def count_nonempty_pixels(ds):
  tot_num_pixels = 0
  tot_num_nonempty_pixels = 0
  for sample in ds:
    if (len(sample) == 3):
      image, label, mask = sample
    else:
      image, label = sample
      mask = None
    assert (image.shape[-1] == 3)
    assert (label.shape[-1] == 1)
    assert (len(image.shape) == 4)
    assert (len(label.shape) == 4)
    # Multiply number of samples, height and width.
    tot_num_pixels_current_batch = (image.shape[0] * image.shape[1] *
                                    image.shape[2])
    tot_num_pixels += tot_num_pixels_current_batch
    if (mask is not None):
      assert (len(mask.shape) == 3)
      tot_num_nonempty_pixels += tf.math.count_nonzero(mask)
    else:
      tot_num_nonempty_pixels += tot_num_pixels_current_batch

  fraction_nonempty_pixels = float(tot_num_nonempty_pixels / tot_num_pixels)

  print("- Fraction non-empty pixels = " +
        "{:.4f}".format(fraction_nonempty_pixels * 100.))
  print("- Total number non-empty pixels = " f"{tot_num_nonempty_pixels}")
  print(f"- Total number pixels = {tot_num_pixels}")

  return tot_num_pixels, tot_num_nonempty_pixels, fraction_nonempty_pixels


dataset_folder = TO_BE_DEFINED

# Change the training dataset with the one from the different data loader.
train_no_replay_ds = load_fsdata(dataset_folder).map(
    preprocess_bagfile_different_dataloader).batch(8).prefetch(
        tf.data.experimental.AUTOTUNE)

count_nonempty_pixels(train_no_replay_ds)
