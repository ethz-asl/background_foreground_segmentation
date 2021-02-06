""" Contains custom callbacks for Keras.
"""
import warnings
from tensorflow.keras.callbacks import Callback


class TestCallback(Callback):
  r"""Callback to evaluate a model on the test set at the end of every epoch.
  """

  def __init__(self, test_data, verbose=0):
    self._test_data = test_data
    self._verbose = verbose

  def on_epoch_end(self, epoch, logs={}):
    self.model.evaluation_type = "test"
    # The metrics need to be manually reset, since no `evaluate` methods is
    # called.
    self.model.logs_test.clear()
    for metric in self.model.metrics:
      metric.reset_states()
    for test_batch in self._test_data:
      logs_test = self.model.test_step(test_batch)
    self.model.logs_test = {k: v.numpy() for k, v in logs_test.items()}
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
      self.model.log_metrics(metric_type="test",
                             logs=self.model.logs_test,
                             step=epoch)