"""
 Main training script file to train for a given experiment
"""
import os, datetime
import json
from sacred import Experiment
from sacred.observers import MongoObserver
import tensorflow as tf
import shutil
from bfseg.utils.evaluation import scoreAndPlotPredictions
from bfseg.experiments.SemSegWithDepthExperiment import SemSegWithDepthExperiment

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
  global args
  args = experiment.getConfig()
  # Generate unique name specifing most of the hyperparameters
  experiment_name = experiment.__class__.__name__ + "{}_{}_{}_{}lr_{}bs_{}ep".format(
      args.name_prefix, args.backbone, args.model_name, args.optimizer_lr,
      args.batch_size, args.num_epochs)
  # Print config and experiment name
  print(experiment_name)
  print(json.dumps(args.__dict__, indent=4, sort_keys=True))

  # Set up sacred experiment
  ex = Experiment(experiment_name)
  ex.observers.append(
      MongoObserver(
          url=
          'mongodb://bfseg_runner:y5jL6uTnHN3W4AZo5oCiG3iX@data.asl.ethz.ch/bfseg',
          db_name='bfseg'))

  # Folder where model checkpoints are stored
  outFolder = os.path.join(args.output_path, experiment_name + "_" + datetime.datetime.now().strftime(
      '%Y%m%d_%H-%M-%S'))
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
    runExperiment(config, experiment, LogMetrics(), outFolder)

  ex.run()


def runExperiment(config, experiment, metricCallback, outFolder):
  """
    Runs a given experiment
    Args:
        config: config
        experiment: assicatied experiment object
        metricCallback: callback used to log metrics to sacred
        outFolder: Where the best weights of this model should be stored

    """

  # Get model from experiment.
  model = experiment.getModel()

  weightsFolder = "./pretrained_nyu_weights_" + config.backbone + "_" + config.model_name
  if isinstance(experiment, SemSegWithDepthExperiment):
      weightsFolder = weightsFolder +  "_with_depth"

  if config.train_from_scratch or not os.path.exists(weightsFolder):
    # pretrain model on nyu data
    pretrainNyu(model, config, weightsFolder, experiment)
  else:
    try:
      model.load_weights(weightsFolder + '/weights.h5')
    except:
      print(
          "Could not load model weights. Starting with random initialized model"
      )
  # Get custom training data from experiment
  train_ds, test_ds = experiment.getTrainData()

  # Score and plot model for pretrained weights. Can also be removed, but shows if our training improves the model
  print("Scoring pretrained model")
  experiment.scoreModel(model, outFolder, tag = "nyu")


  print("Training model")
  # Compile model to be trained on training dataset
  experiment.compileModel(model)

  callbacks = [
      # tf.keras.callbacks.ModelCheckpoint('./' + outFolder +
      #                                    '/model.{epoch:02d}-{val_loss:.2f}.h5',
      #                                    save_weights_only=True,
      #                                    save_best_only=True,
      #                                    mode='min'),
      # Log metrics to sacred
      metricCallback
  ]


  model.fit(train_ds,
            epochs=config.num_epochs,
            # validation_data=test_ds,
            callbacks=callbacks)
  model.save(outFolder + "/model.h5")

  experiment.scoreModel(model, outFolder, exportImages = True, tag = "vicon")
  print("Scored model. Finished")


def pretrainNyu(model, config, weightsFolder, experiment):
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
  train_nyu_ds, valid_nyu_ds, steps_per_epoch = experiment.getNyuTrainData()
  # Compile model to train on nyu (This can be differ e.g. learning rate higher, different loss)
  experiment.compileNyuModel(model)

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
            epochs=config.nyu_epochs,
            validation_data=valid_nyu_ds,
            callbacks=callbacks)
