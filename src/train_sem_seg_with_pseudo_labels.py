"""
Entry point to train with pseudo labels and meshdist labels
"""
import bfseg.experiments.SemSegWithPseudoLabels as experimentWrapper
from train_experiment import train

if __name__ == "__main__":
    train(experimentWrapper.SemSegWithPseudoLabelsExperiment())