""" Contains custom callbacks for Keras.
"""
import warnings
from tensorflow.keras.callbacks import Callback, EarlyStopping


class TestCallback(Callback):
  r"""Callback to evaluate a model on the test set at the end of every epoch.

  Args:
    test_data (tensorflow.python.data.ops.dataset_ops.PrefetchDataset/dict):
      Either single test dataset to use, or dictionary of the multiple test
      datasets to use, indexed by their name.
    verbose (int): Verbose level.
  """

  def __init__(self, test_data, verbose=0):
    if (isinstance(test_data, dict)):
      self._test_data = test_data
    else:
      # If single test dataset is given, still convert it to dict format, for
      # compatibility with the multiple-dataset case.
      self._test_data = {'test': test_data}

    self._verbose = verbose

  def on_epoch_end(self, epoch, logs={}):
    self.model.evaluation_type = "test"
    # The metrics need to be manually reset, since no `evaluate` methods is
    # called.
    self.model.logs_test.clear()
    for test_dataset_name, test_dataset in self._test_data.items():
      for metric in self.model.metrics:
        metric.reset_states()
      for test_batch in test_dataset:
        logs_test = self.model.test_step(test_batch)
      self.model.logs_test[test_dataset_name] = {
          k: v.numpy() for k, v in logs_test.items()
      }
    self.model.performed_test_evaluation = True
    self.model.evaluation_type = "val"


class SaveModelAndLogs(Callback):

  def __init__(self):
    pass

  def on_epoch_end(self, epoch, logs={}):
    # Optionally save the model at the end of the current epoch.
    if ((epoch + 1) %
        self.model.run.config['logging_params']['model_save_freq'] == 0):
      self.model.save_model(epoch=epoch)
    # Log the metrics.
    # - Split training and validation logs.
    logs_train = {}
    logs_val = {}
    for metric_name, metric_value in logs.items():
      if (metric_name[:4] == "val_"):
        # Remove the prefix `val_` from the validation metrics.
        logs_val[metric_name[4:]] = metric_value
      else:
        logs_train[metric_name] = metric_value

    self.model.log_metrics(metric_type='train', logs=logs_train, step=epoch)
    # Log metrics for validation and test set.
    self.model.log_metrics(metric_type="val", logs=logs_val, step=epoch)
    if (not self.model.performed_test_evaluation):
      warnings.warn(
          "Did not perform test evaluation at the end of this epoch. If you "
          "have set a test evaluation to happen and wish it to be logged, make "
          "sure to call the TestCallback before this callback.")
    else:
      for (test_dataset_name,
           test_dataset_logs) in self.model.logs_test.items():
        self.model.log_metrics(metric_type=test_dataset_name,
                               logs=test_dataset_logs,
                               step=epoch)


class EarlyStoppingMinimumEpoch(EarlyStopping):
  r"""Performs the usual early-stopping, but starts it at an epoch such that
  when early stopping happens at least `min_epoch` epochs have been performed.
  """

  def __init__(self, min_epoch, **kwargs):
    super(EarlyStoppingMinimumEpoch, self).__init__(**kwargs)
    self._start_epoch = max(0, min_epoch - self.patience)

  def on_epoch_end(self, epoch, logs=None):
    if (epoch > self._start_epoch):
      super().on_epoch_end(epoch, logs)
