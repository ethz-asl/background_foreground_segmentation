import argparse
from bfseg.utils.evaluation import scoreAndPlotPredictions
import os
import json
from sacred import Experiment as SacredExperiment
from sacred.observers import MongoObserver
import tensorflow as tf
import shutil
import datetime


class Experiment():
  """
    Base class to specify a expiriment.
    An experiment is a standalone class that supports:
    - Loading training data
    - Creating Models to train
    - Compiling models with experiment specific loss functions (e.g. pseudo labels loss)
    - Training the model

    Each experiment can register custom arguments by overwriting the _addArguments() function.
    """

  def __init__(self):
    self.config = self.getConfig()
    self.weightsFolder = "./pretrained_nyu_weights_" + self.config.backbone + "_" + self.config.model_name

  def runExperiment(self, metricCallback, outFolder):
    """
          Runs a given experiment
          Args:
              metricCallback: callback used to log metrics to sacred
              outFolder: Where the weights and predictions of this model should be stored
          """

    model = self.getModel()

    if self.config.train_from_scratch or not os.path.exists(self.weightsFolder):
      # pretrain model on nyu data
      self.pretrainNyu(model, self.weightsFolder)
    else:
      try:
        model.load_weights(weightsFolder + '/weights.h5')
      except Exception as e:
        print("Could not load pretrained weights. Starting with random ones.",
              e)

    # Get custom training data from experiment
    train_ds, test_ds = self.getTrainData()

    # Score and plot model for pretrained weights. Can also be removed, but shows if our training improves the model
    print("Scoring pretrained model")
    self.scoreModel(model, outFolder, tag="nyu")

    print("Training model")
    # Compile model to be trained on training dataset
    self.compileModel(model)
    callbacks = [metricCallback]
    model.fit(train_ds,
              epochs=self.config.num_epochs,
              validation_data=test_ds,
              callbacks=callbacks)

    # Save final mode
    model.save(outFolder + "/model.h5")
    # Export predictions + results
    self.scoreModel(model, outFolder, exportImages=True, tag="vicon")
    print("Scored model. Finished")

  def pretrainNyu(self, model, weightsFolder):
    """
        Pretrains a given model on the nyu dataset and stores the pretrained weight in the weights folder
        Args:
            model: model that supports .fit() function
            config: config
            weightsFolder: folder where the weights should be stored
            experiment: Associated Experiment class
        """

    print("Pretraining on NYU")
    # Get training data from experiment
    train_nyu_ds, valid_nyu_ds, steps_per_epoch = self.getNyuTrainData()
    # Compile model to train on nyu (This can be differ e.g. learning rate higher, different loss)
    self.compileNyuModel(model)

    if os.path.exists(weightsFolder):
      print("found old weight. Going to remove it")
      shutil.rmtree(weightsFolder)

    os.mkdir(weightsFolder)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(weightsFolder + '/weights.h5',
                                           save_weights_only=True,
                                           save_best_only=True,
                                           mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau()
    ]

    model.fit(train_nyu_ds,
              steps_per_epoch=steps_per_epoch,
              epochs=self.config.nyu_epochs,
              validation_data=valid_nyu_ds,
              callbacks=callbacks)

  def getConfig(self):
    """ Loads config from argparser """
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve")
    self._addArguments(parser)

    return parser.parse_args()

  def _addArguments(self, parser):
    """ Function used to add custom parameters for the experiment. Base one only has a name prefix"""
    parser.add_argument('--name_prefix',
                        type=str,
                        help='Name Prefix',
                        default="")
    parser.add_argument('--output_path',
                        type=str,
                        help='Output Path',
                        default="")

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

  def scoreModel(self, model, test_ds):
    scoreAndPlotPredictions(lambda img: model.predict(img),
                            test_ds,
                            self.numTestImages,
                            plot=False)


def train(experiment):
  """
    Main entry point to train for a given experiment
    Args:
        experiment: Class inherited from Experiment base class.
                    Needs to provide the following functionalities:
                    - getNyuTrainData(self)   <- Get NYU training data (Used to pretrain model)
                    - getTrainData()          <- get Training data
                    - getModel()              <- get model that supports .fit() function
                    - compileModel(model)
                    - compileNyuModel(model)
    """
  # needs to be global to be set using @ex.config
  global args, outFolder, experiment_name
  args = experiment.getConfig()
  # Generate unique name specifing most of the hyperparameters
  experiment_name = experiment.__class__.__name__ + "{}_{}_{}_{}lr_{}bs_{}ep".format(
      args.name_prefix, args.backbone, args.model_name, args.optimizer_lr,
      args.batch_size, args.num_epochs)

  # Set up sacred experiment
  ex = SacredExperiment(experiment_name)
  ex.observers.append(
      MongoObserver(
          url=
          'mongodb://bfseg_runner:y5jL6uTnHN3W4AZo5oCiG3iX@data.asl.ethz.ch/bfseg',
          db_name='bfseg'))

  # Folder where model checkpoints are stored
  outFolder = os.path.join(
      args.output_path, experiment_name + "_" +
      datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S'))
  os.mkdir(outFolder)

  class LogMetrics(tf.keras.callbacks.Callback):
    """
            Callback to log metrics to sacred
        """

    def on_epoch_end(self, _, logs={}):
      my_metrics(logs=logs)

  @ex.capture
  def my_metrics(_run, logs):
    """ log all metrics to sacred"""
    for key, value in logs.items():
      _run.log_scalar(key, float(value))

  @ex.config
  def cfg():
    """ Wrapper to convert arguments from console to sacred config"""
    config = args

  @ex.main
  def run(config):
    # Print config and experiment name
    print(experiment_name)
    print(json.dumps(args.__dict__, indent=4, sort_keys=True))
    experiment.runExperiment(LogMetrics(), outFolder)

  ex.run()
