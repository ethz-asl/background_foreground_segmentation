"""
Entry point to only train with meshdist labels
"""

import bfseg.experiments.TimestampBasedExperiments as experimentWrapper
from bfseg.experiments.Experiment import train

if __name__ == "__main__":
  train(experimentWrapper.TimestampBasedSemSegExperiment())
