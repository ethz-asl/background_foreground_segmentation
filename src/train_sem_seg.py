"""
Entry point to only train with meshdist labels
"""

import bfseg.experiments.SemSegExperiment as experimentWrapper
from train_experiment import train

if __name__ == "__main__":
  train(experimentWrapper.SemSegExperiment())
