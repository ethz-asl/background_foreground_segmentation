""" Contains custom callbacks for Keras.
"""
import warnings
from tensorflow.keras import Callback


class TestCallback(Callback):
  r"""Callback to evaluate a model on the test set at the end of every epoch.
    Adapted from https://github.com/keras-team/keras/issues/
    2548#issuecomment-215664770.
    """

  def __init__(self, test_data, verbose=0):
    self._test_data = test_data
    self._verbose = verbose

  def on_epoch_end(self, epoch, logs={}):
    self.model.evalation_type = "test"
    self.model.evaluate(self._data, verbose=self._verbose)
    self.model.performed_test_evaluation = True
    self.model.evalation_type = "val"


class SaveModelAndLogs(Callback):

  def __init__(self):
    pass

  def on_epoch_end(self, epoch, logs={}):
    # Optionally save the model at the end of the current epoch.
    if ((epoch + 1) %
        self.model.run.config['logging_params']['model_save_freq'] == 0):
      self.model.save_model(epoch=epoch)
    # Optionally log the metrics.
    if (self.model.metric_log_frequency == "epoch"):
      self.model.log_metrics(metric_type='train', step=epoch)
      val_test_logging_step = epoch
    else:
      val_test_logging_step = self.model.current_batch
    # Log metrics for validation and test set.
    self.model.log_metrics("val", step=val_test_logging_step)
    # Evaluate on test set.
    if (not self.model.performed_test_evaluation):
      warnings.warn(
          "Did not perform test evaluation at the end of this epoch. If you "
          "have set a test evaluation to happen and wish it to be logged, make "
          "sure to call the TestCallback before this callback.")
    else:
      self.model.log_metrics("test", step=val_test_logging_step)
