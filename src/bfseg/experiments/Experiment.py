import argparse

class Experiment():
    """
        Base class to specify a expiriment.
        An experiment is a standalone class that supports:
        - Loading training data
        - Creating Models to train
        - Compiling models with experiment specific loss functions (e.g. pseudo labels loss)

        Each experiment can register custom arguments by overwriting the _addArguments() function.
    """

    def __init__(self):
        self.config = self.getConfig();

    def getConfig(self):
        """
            Load config from argparser
        """
        parser = argparse.ArgumentParser(
            add_help=True,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self._addArguments(parser)

        return parser.parse_args()

    def _addArguments(self, parser):
        """ Function used to add custom parameters for the experiment. Base one only has a name prefix"""
        parser.add_argument(
            '--name_prefix', type=str, help='Name Prefix', default="")

    def getNyuTrainData(self):
        """
            Should return a train and test ds with nyu data. (Tensorflow datasets)
            If the experiment is not pretrainable on NYU it should raise a NotImplementedError and the training
            will be skipped
        """
        raise NotImplementedError()

    def getTrainData(self):
        """
        Returns: a train_ds and test_ds to train the model on the real experiment data. (Tensorflow datasets)
        """
        raise NotImplementedError()

    def getModel(self):
        """
        Returns: a model that can be trained with the training data for this experiment.
        """
        raise NotImplementedError()

    def compileModel(self, model):
        """
        Compiles a movel with a experiment specific loss and metric function
        """
        raise NotImplementedError()

    def compileNyuModel(self, model):
        """
        Compile the model with a nyu specific loss and metric function
        """
        raise NotImplementedError()
