import os, datetime
import json
from sacred import Experiment
from sacred.observers import MongoObserver
import tensorflow as tf
# Uncomment to disable GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["SM_FRAMEWORK"] = "tf.keras"

from bfseg.experiments import SemSegExperiment as experimentWrapper

experiment = experimentWrapper.SemSegExpirement()

args = experiment.getConfig()
experiment_name = "{}_{}_{}_{}lr_{}bs_{}ep".format(args.name_prefix, args.backbone, args.model_name,
                                                   args.optimizer_lr, args.batch_size, args.num_epochs)
print(experiment_name)
print(json.dumps(args.__dict__, indent=4, sort_keys=True))

# Set up sacred experiment
ex = Experiment(experiment_name)
ex.observers.append(
    MongoObserver(
        url=
        'mongodb://bfseg_runner:y5jL6uTnHN3W4AZo5oCiG3iX@data.asl.ethz.ch/bfseg',
        db_name='bfseg'))

outFolder = experiment_name + "_" + datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')
os.mkdir(experiment_name + "_" + datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S'))


# Tweak GPU settings for local use
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# workingdir = "/cluster/scratch/zrene/cla_dataset/watershed/"
# validationDir = '/cluster/scratch/zrene/cla_dataset/hiveLabels/'
# baselinePath = "./baseline_model.h5"
#
# try:
#   if os.environ['local']:
#     workingdir = "/home/rene/cla_dataset/watershed/"
#     validationDir = '/home/rene/hiveLabels/'
# except:
#   print("Running on cluster")

# Desired image shape. Input images will be cropped + scaled to this shape

class LogMetrics(tf.keras.callbacks.Callback):
    """
    Logs the metrics of the current epoch in tensorboard
    """

    def on_epoch_end(self, _, logs={}):
        my_metrics(logs=logs)


@ex.capture
def my_metrics(_run, logs):
    for key, value in logs.items():
        _run.log_scalar(key, float(value))


@ex.config
def cfg():
    config = args


def pretrainNyu(model, config, weightsFolder):
    print("Pretraining on NYU")
    # pretrain model on Nyu dataset
    train_nyu_ds, valid_nyu_ds = experiment.getNyuTrainData()
    experiment.compileNyuModel(model)

    os.mkdir(weightsFolder)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(weightsFolder + '/weights.h5',
                                           save_weights_only=True,
                                           save_best_only=True,
                                           mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau()
    ]

    model.train(train_nyu_ds,
                epochs=config.nyu_epochs,
                validation_data=valid_nyu_ds,
                callbacks=callbacks)



@ex.main
def run(config):
    model = experiment.getModel()

    weightsFolder = "./pretrained_nyu_weights_" + config.backbone + "_" + config.model_name

    if config.train_from_scratch:
      pretrainNyu(model,config,weightsFolder)
    else:
        try:
            model.load_weights(weightsFolder + '/weights.h5')
        except:
            print(
                "Could not load model weights. Starting with random initialized model"
            )

    train_ds, test_ds = experiment.getTrainData()
    print(test_ds.__dict__)
    print("Scoring pretrained model")
    from bfseg.utils.evaluation import scoreAndPlotPredictions
    scoreAndPlotPredictions(lambda img: model.predict(img), test_ds,  plot=False)

    print("Training model")

    experiment.compileModel(model)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            './' + outFolder + '/model.{epoch:02d}-{val_loss:.2f}.h5',
            save_weights_only=True,
            save_best_only=True,
            mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(),
        # Log metrics to sacred
        LogMetrics()
    ]

    model.fit(train_ds,
              epochs=config.num_epochs,
              validation_data=test_ds,
              callbacks=callbacks)


if __name__ == "__main__":
    ex.run()
# input("Press enter to stop")
