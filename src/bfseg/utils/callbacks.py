""" Contains custom callbacks for Keras.
"""

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
    loss, acc = self.model.evaluate(self._data, verbose=self._verbose)
    self.model.evalation_type = "val"
