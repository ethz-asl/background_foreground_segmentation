"""
Entry point to only train with meshdist labels
"""

import bfseg.experiments.SemSegWithDepthExperiment as experimentWrapper
from train_experiment import train

if __name__ == "__main__":
  train(experimentWrapper.SemSegWithDepthExperiment())