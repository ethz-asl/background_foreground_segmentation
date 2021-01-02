"""
Entry point to only train_experiments with meshdist labels
"""

import bfseg.experiments.SemSegWithDepthExperiment as experimentWrapper
from bfseg.experiments.Experiment import train

if __name__ == "__main__":
  train(experimentWrapper.SemSegWithDepthExperiment())
