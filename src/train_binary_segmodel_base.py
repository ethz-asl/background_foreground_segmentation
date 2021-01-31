import sys
sys.path.append("..")
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import argparse
import datetime
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import bfseg.data.nyu.Nyu_depth_v2_labeled
import bfseg.data.meshdist.bfseg_cla_meshdist_labels
from bfseg.sacred_utils import get_observer
from bfseg.settings import TMPDIR
from sacred import Experiment
import segmentation_models as sm

ex = Experiment()
ex.observers.append(get_observer())


class BaseSegExperiment():
  """
    Base class to specify an experiment.
    An experiment is a standalone class that supports:
    - Loading training data
    - Creating Models that can be trained
    - Creating experiment specific loss functions
    - Creating tensorboard writer for visualization
    - Training the model
    """

  def __init__(self):
    self.config = self.getConfig()
    self.log_dir = os.path.join(self.config.exp_dir, self.config.exp_name,
                                'logs')
    self.model_save_dir = os.path.join(self.config.exp_dir,
                                       self.config.exp_name, 'saved_model')
    self.optimizer = keras.optimizers.Adam(self.config.lr)

  def make_dirs(self):
    try:
      os.makedirs(self.log_dir + '/train')
      os.makedirs(self.log_dir + '/val')
      os.makedirs(self.log_dir + '/test')
      os.makedirs(self.model_save_dir)
    except os.error:
      pass

  def _addArguments(self, parser):
    """ Function used to add custom parameters for the experiment."""
    parser.add_argument('-batch_size', default=8, type=int, help='batch size')
    parser.add_argument('-epochs',
                        default=3,
                        type=int,
                        help='number of training epoches')
    parser.add_argument('-image_w', default=640, type=int, help='image width')
    parser.add_argument('-image_h', default=480, type=int, help='image height')
    parser.add_argument('-exp_dir',
                        default='../experiments',
                        type=str,
                        help='directory of current experiments')
    parser.add_argument('-exp_name',
                        default="exp_stage1",
                        type=str,
                        help='name of current experiments')
    parser.add_argument('-backbone',
                        default="vgg16",
                        type=str,
                        help='name of backbone')
    parser.add_argument('-data_dir',
                        default="../tensorflow_datasets",
                        type=str,
                        help='directory of dataset')
    parser.add_argument('-lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('-train_dataset',
                        default="BfsegCLAMeshdistLabels",
                        type=str,
                        help='name of training dataset')
    parser.add_argument('-test_dataset',
                        default="NyuDepthV2Labeled",
                        type=str,
                        help='name of testing dataset')
    parser.add_argument(
        '-train_scene',
        default=None,
        type=str,
        help='scene type of training dataset: None/kitchen/bedroom')
    parser.add_argument(
        '-test_scene',
        default=None,
        type=str,
        help='scene type of testing dataset: None/kitchen/bedroom')
    parser.add_argument('-pretrained_dir',
                        default=None,
                        type=str,
                        help='directory of pretrained model weights')
    parser.add_argument('-tensorboard_write_freq',
                        default="batch",
                        type=str,
                        help='write to tensorboard per epoch/batch')
    parser.add_argument('-model_save_freq',
                        default=1,
                        type=int,
                        help='frequency of saving models (epochs)')

  def getConfig(self):
    """ Loads config from argparser """
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve")
    self._addArguments(parser)
    return parser.parse_args()

  @tf.function
  def preprocess_nyu(self, image, label):
    """ Preprocess NYU dataset:
            Normalize images: `uint8` -> `float32`.
            label: 1 if belong to background, 0 if foreground
            create all-True mask since nyu labels are all known
        """
    mask = tf.not_equal(label, -1)  #all true
    label = tf.expand_dims(label, axis=2)
    image = tf.cast(image, tf.float32) / 255.
    return image, label, mask

  @tf.function
  def preprocess_cla(self, image, label):
    """ Preprocess our auto-labeled CLA dataset:
            It consists of three labels (0,1,2) where all classes that belong to the background 
            (e.g. floor, wall, roof) are assigned the '2' label. Foreground has assigned the 
            '0' label and unknown the '1' label,
            Let CLA label format to be consistent with NYU
            label: 1 if belong to background, 0 if foreground / unknown(does not matter since we are using masked loss)
            Mask element is True if it's known (label '0' or '2'), mask is used to compute masked loss
        """
    mask = tf.squeeze(tf.not_equal(label, 1))
    label = tf.cast(label == 2, tf.uint8)
    image = tf.cast(image, tf.float32)
    return image, label, mask

  def load_data(self, dataset, data_dir, mode, batch_size, scene_type):
    """ Create a dataloader given:
            name of dataset: NyuDepthV2Labeled/ BfsegCLAMeshdistLabels,
            mode: train/ val/ test
            type of scene: None/ kitchen/ bedroom
        """
    if dataset == 'NyuDepthV2Labeled':
      if scene_type == None:
        name = 'full'
      elif scene_type == "kitchen":
        name = 'train'
      elif scene_type == "bedroom":
        name = 'test'
      else:
        raise Exception("Invalid scene type: %s!" % scene_type)
    elif dataset == 'BfsegCLAMeshdistLabels':
      name = 'fused'
    else:
      raise Exception("Dataset %s not found!" % dataset)
    if mode == 'train':
      split = name + '[:80%]'
      shuffle = True
    else:
      split = name + '[80%:]'
      shuffle = False
    ds, info = tfds.load(
        dataset,
        split=split,
        data_dir=data_dir,
        shuffle_files=shuffle,
        as_supervised=True,
        with_info=True,
    )
    if dataset == 'NyuDepthV2Labeled':
      ds = ds.map(self.preprocess_nyu,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif dataset == 'BfsegCLAMeshdistLabels':
      ds = ds.map(self.preprocess_cla,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    if mode == 'train':
      ds = ds.shuffle(int(info.splits[name].num_examples * 0.8))
    ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def load_dataset(self, train_dataset, train_scene, test_dataset, test_scene,
                   data_dir, batch_size):
    """create 3 dataloaders for training, validation and testing """
    train_ds = self.load_data(train_dataset, data_dir, 'train', batch_size,
                              train_scene)
    val_ds = self.load_data(train_dataset, data_dir, 'val', batch_size,
                            train_scene)
    test_ds = self.load_data(test_dataset, data_dir, 'test', batch_size,
                             test_scene)
    return train_ds, val_ds, test_ds

  def create_old_params(self):
    """ Keep old weights of the model"""
    pass

  def create_fisher_params(self, dataset):
    """ Compute sqaured fisher information, representing relative importance"""
    pass

  def compute_consolidation_loss(self):
    """ Compute weight regularization term """
    pass

  def build_model(self):
    """ Build models"""
    self.encoder, self.model = sm.Unet(self.config.backbone,
                                       input_shape=(self.config.image_h,
                                                    self.config.image_w, 3),
                                       classes=2,
                                       activation='sigmoid',
                                       weights=self.config.pretrained_dir,
                                       encoder_freeze=False)
    self.new_model = keras.Model(
        inputs=self.model.input,
        outputs=[self.encoder.output, self.model.output])

  def build_tensorboard_writer(self):
    """ Create tensorboard writers"""
    self.train_summary_writer = tf.summary.create_file_writer(self.log_dir +
                                                              "/train")
    self.val_summary_writer = tf.summary.create_file_writer(self.log_dir +
                                                            "/val")
    self.test_summary_writer = tf.summary.create_file_writer(self.log_dir +
                                                             "/test")

  def build_loss_and_metric(self):
    """ Add loss criteria and metrics"""
    self.loss_ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.loss_tracker = keras.metrics.Mean('loss', dtype=tf.float32)
    self.acc_metric = keras.metrics.Accuracy('accuracy')

  def train_step(self, train_x, train_y, train_m, step):
    """ Training on one batch:
            Compute masked cross entropy loss(true label, predicted label),
            update losses & metrics
        """
    with tf.GradientTape() as tape:
      [_, pred_y] = self.new_model(train_x, training=True)
      pred_y_masked = tf.boolean_mask(pred_y, train_m)
      train_y_masked = tf.boolean_mask(train_y, train_m)
      loss = self.loss_ce(train_y_masked, pred_y_masked)
    grads = tape.gradient(loss, self.new_model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.new_model.trainable_weights))
    pred_y = tf.math.argmax(pred_y, axis=-1)
    pred_y_masked = tf.boolean_mask(pred_y, train_m)
    self.acc_metric.update_state(train_y_masked, pred_y_masked)

    if self.config.tensorboard_write_freq == "batch":
      with self.train_summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.scalar('accuracy', self.acc_metric.result(), step=step)
      self.acc_metric.reset_states()
    elif self.config.tensorboard_write_freq == "epoch":
      self.loss_tracker.update_state(loss)
    else:
      raise Exception("Invalid tensorboard_write_freq: %s!" %
                      self.config.tensorboard_write_freq)

  def test_step(self, test_x, test_y, test_m):
    """ Validating/Testing on one batch
            update losses & metrics
        """
    [_, pred_y] = self.new_model(test_x, training=False)
    pred_y_masked = tf.boolean_mask(pred_y, test_m)
    test_y_masked = tf.boolean_mask(test_y, test_m)
    loss = self.loss_ce(test_y_masked, pred_y_masked)
    pred_y = keras.backend.argmax(pred_y, axis=-1)
    pred_y_masked = tf.boolean_mask(pred_y, test_m)
    # Update val/test metrics
    self.loss_tracker.update_state(loss)
    self.acc_metric.update_state(test_y_masked, pred_y_masked)

  def on_epoch_end(self, epoch, val_ds):
    """ save models after every several epochs
        """
    if (epoch + 1) % self.config.model_save_freq == 0:
      # compute validation accuracy as part of the model name
      for val_x, val_y, val_m in val_ds:
        self.test_step(val_x, val_y, val_m)
      self.new_model.save(
          os.path.join(
              self.model_save_dir, 'model.' + str(epoch) + '-' +
              str(self.acc_metric.result().numpy())[:5] + '.h5'))
      self.loss_tracker.reset_states()
      self.acc_metric.reset_states()

  def write_to_tensorboard(self, summary_writer, step):
    """
            write losses and metrics to tensorboard
        """
    with summary_writer.as_default():
      tf.summary.scalar('loss', self.loss_tracker.result(), step=step)
      tf.summary.scalar('accuracy', self.acc_metric.result(), step=step)
    self.loss_tracker.reset_states()
    self.acc_metric.reset_states()

  def training(self, train_ds, val_ds, test_ds):
    """
        train for assigned epochs, and validate & test after each epoch/batch,
        save models after every several epochs
        """
    step = 0
    for epoch in range(self.config.epochs):
      print("\nStart of epoch %d" % (epoch,))
      if self.config.tensorboard_write_freq == "batch":
        for train_x, train_y, train_m in train_ds:
          self.train_step(train_x, train_y, train_m, step)
          for val_x, val_y, val_m in val_ds:
            self.test_step(val_x, val_y, val_m)
          self.write_to_tensorboard(self.val_summary_writer, step)
          for test_x, test_y, test_m in test_ds:
            self.test_step(test_x, test_y, test_m)
          self.write_to_tensorboard(self.test_summary_writer, step)
          step += 1
        if epoch == 0:
          print("There are %d batches in the training dataset" % step)
      elif self.config.tensorboard_write_freq == "epoch":
        for train_x, train_y, train_m in train_ds:
          self.train_step(train_x, train_y, train_m, step)
        self.write_to_tensorboard(self.train_summary_writer, epoch)
        for val_x, val_y, val_m in val_ds:
          self.test_step(val_x, val_y, val_m)
        self.write_to_tensorboard(self.val_summary_writer, epoch)
        for test_x, test_y, test_m in test_ds:
          self.test_step(test_x, test_y, test_m)
        self.write_to_tensorboard(self.test_summary_writer, epoch)
      self.on_epoch_end(epoch, val_ds)


@ex.main
def run(_run):
  """ Whole Training pipeline"""
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  print("Current time is " + current_time)
  seg_experiment = BaseSegExperiment()
  seg_experiment.make_dirs()
  seg_experiment.build_model()
  seg_experiment.build_tensorboard_writer()
  seg_experiment.build_loss_and_metric()
  train_ds, val_ds, test_ds = seg_experiment.load_dataset(
      seg_experiment.config.train_dataset, seg_experiment.config.train_scene,
      seg_experiment.config.test_dataset, seg_experiment.config.test_scene,
      seg_experiment.config.data_dir, seg_experiment.config.batch_size)
  seg_experiment.training(train_ds, val_ds, test_ds)


if __name__ == "__main__":
  ex.run_commandline()
