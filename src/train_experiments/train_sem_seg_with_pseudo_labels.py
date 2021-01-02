"""
Entry point to train_experiments with pseudo labels and meshdist labels
"""
import bfseg.experiments.SemSegWithPseudoLabels as experimentWrapper
from bfseg.experiments.Experiment import train

if __name__ == "__main__":
  train(experimentWrapper.SemSegWithPseudoLabelsExperiment())
