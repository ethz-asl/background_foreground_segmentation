import sys
sys.path.append("..")
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from bfseg.utils.datasets import load_data
import datetime
import tensorflow as tf
from tensorflow import keras
from train_binary_segmodel_base import BaseSegExperiment


class EWC(BaseSegExperiment):
  """
    Experiment to train on 2nd task with EWC algorithm 
    "Overcoming catastrophic forgetting in neural networks"
    https://arxiv.org/abs/1612.00796
    """

  def __init__(self):
    super(EWC, self).__init__()
    print("lambda is " + str(self.config.lambda_weights))
    self.log_dir = os.path.join('../experiments', self.config.exp_name,
                                'lambda' + str(self.config.lambda_weights),
                                'logs')
    self.model_save_dir = os.path.join(
        '../experiments', self.config.exp_name,
        'lambda' + str(self.config.lambda_weights), 'saved_model')

  def _addArguments(self, parser):
    """ Add custom arguments that are needed for this experiment """
    super(EWC, self)._addArguments(parser)
    parser.add_argument(
        '-lambda_weights',
        default=0,
        type=float,
        help=
        'weighting constant, range: [0,1], lambda=0: no weight constraints, lambda=1: no train on 2nd task'
    )

  def load_dataset(self, train_dataset, train_scene, test_dataset, test_scene,
                   data_dir, batch_size):
    """ Create 3 dataloaders for training, validation and testing """
    train_ds = load_data(train_dataset, data_dir, 'train', batch_size,
                         train_scene)
    val_ds = load_data(train_dataset, data_dir, 'val', batch_size, train_scene)
    test_ds = load_data(test_dataset, data_dir, 'test', batch_size, test_scene)
    pretrain_ds = load_data(test_dataset, data_dir, 'train', batch_size,
                            test_scene)
    return train_ds, val_ds, test_ds, pretrain_ds

  def create_old_params(self):
    """ Keep old weights of the model"""
    self.old_params = []
    for param in self.new_model.trainable_weights:
      old_param_name = param.name.replace(':0', '_old')
      self.old_params.append(
          tf.Variable(param, trainable=False, name=old_param_name))

  def create_fisher_params(self, dataset):
    """ Compute sqaured fisher information, representing relative importance"""
    self.fisher_params = []
    grads_list = [
    ]  # list of list of gradients, outer: for different batches, inner: for different network parameters
    for step, (x, y, m) in enumerate(dataset):
      if step > 40:
        break
      with tf.GradientTape() as tape:
        [_, pred_y] = self.new_model(x, training=True)
        log_y = tf.math.log(pred_y)
        y = tf.cast(y, log_y.dtype)
        log_likelihood = tf.reduce_sum(y * log_y[:, :, :, 1:2] +
                                       (1 - y) * log_y[:, :, :, 0:1],
                                       axis=[1, 2, 3])
      grads = tape.gradient(log_likelihood, self.new_model.trainable_weights)
      grads_list.append(grads)
    fisher_params = []
    fisher_param_names = [
        param.name.replace(':0', '_fisher')
        for param in self.new_model.trainable_weights
    ]
    ## compute expectation
    for i in range(len(fisher_param_names)):
      single_fisher_param_list = [tf.square(param[i]) for param in grads_list]
      fisher_params.append(
          tf.reduce_mean(tf.stack(single_fisher_param_list, 0), 0))
    for param_name, param in zip(fisher_param_names, fisher_params):
      self.fisher_params.append(
          tf.Variable(param, trainable=False, name=param_name))

  def compute_consolidation_loss(self):
    """ Compute weight regularization term """
    losses = []
    for i, param in enumerate(self.new_model.trainable_weights):
      losses.append(
          tf.reduce_sum(self.fisher_params[i] *
                        (param - self.old_params[i])**2))
    return tf.reduce_sum(losses)

  def build_loss_and_metric(self):
    """ Add loss criteria and metrics"""
    self.loss_ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.loss_mse = keras.losses.MeanSquaredError()
    self.loss_tracker = keras.metrics.Mean('loss', dtype=tf.float32)
    self.loss_ce_tracker = keras.metrics.Mean('loss_ce', dtype=tf.float32)
    self.loss_mse_tracker = keras.metrics.Mean('loss_weights', dtype=tf.float32)
    self.acc_metric = keras.metrics.Accuracy('accuracy')

  def train_step(self, train_x, train_y, train_m, step):
    """ Training on one batch:
            Compute masked cross entropy loss(true label, predicted label)
            + Weight consolidation loss(new weights, old weights)
            update losses & metrics
        """
    with tf.GradientTape() as tape:
      [_, pred_y] = self.new_model(train_x, training=True)
      pred_y_masked = tf.boolean_mask(pred_y, train_m)
      train_y_masked = tf.boolean_mask(train_y, train_m)
      output_loss = self.loss_ce(train_y_masked, pred_y_masked)
      consolidation_loss = self.compute_consolidation_loss()
      loss = (1 - self.config.lambda_weights
             ) * output_loss + self.config.lambda_weights * consolidation_loss
    grads = tape.gradient(loss, self.new_model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.new_model.trainable_weights))
    pred_y = tf.math.argmax(pred_y, axis=-1)
    pred_y_masked = tf.boolean_mask(pred_y, train_m)
    self.acc_metric.update_state(train_y_masked, pred_y_masked)

    if self.config.tensorboard_write_freq == "batch":
      with self.train_summary_writer.as_default():
        tf.summary.scalar('loss,lambda=' + str(self.config.lambda_weights),
                          loss,
                          step=step)
        tf.summary.scalar('loss_ce,lambda=' + str(self.config.lambda_weights),
                          output_loss,
                          step=step)
        tf.summary.scalar('loss_weights,lambda=' +
                          str(self.config.lambda_weights),
                          consolidation_loss,
                          step=step)
        tf.summary.scalar('accuracy,lambda=' + str(self.config.lambda_weights),
                          self.acc_metric.result(),
                          step=step)
      self.acc_metric.reset_states()
    elif self.config.tensorboard_write_freq == "epoch":
      self.loss_tracker.update_state(loss)
      self.loss_ce_tracker.update_state(output_loss)
      self.loss_mse_tracker.update_state(consolidation_loss)
    else:
      raise Exception("Invalid tensorboard_write_freq: %s!" %
                      self.config.tensorboard_write_freq)

  def test_step(self, test_x, test_y, test_m):
    """ Validating/Testing on one batch
            update losses & metrics
        """
    [_, pred_y] = self.new_model(test_x, training=True)
    pred_y_masked = tf.boolean_mask(pred_y, test_m)
    test_y_masked = tf.boolean_mask(test_y, test_m)
    output_loss = self.loss_ce(test_y_masked, pred_y_masked)
    consolidation_loss = self.compute_consolidation_loss()
    loss = (1 - self.config.lambda_weights
           ) * output_loss + self.config.lambda_weights * consolidation_loss
    pred_y = tf.math.argmax(pred_y, axis=-1)
    pred_y_masked = tf.boolean_mask(pred_y, test_m)
    # Update val/test metrics
    self.loss_tracker.update_state(loss)
    self.loss_ce_tracker.update_state(output_loss)
    self.loss_mse_tracker.update_state(consolidation_loss)
    self.acc_metric.update_state(test_y_masked, pred_y_masked)

  def on_epoch_end(self, epoch, val_ds):
    """save models after every several epochs
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
      self.loss_ce_tracker.reset_states()
      self.loss_mse_tracker.reset_states()
      self.acc_metric.reset_states()

  def write_to_tensorboard(self, summary_writer, step):
    """
            write losses and metrics to tensorboard
        """
    with summary_writer.as_default():
      tf.summary.scalar('loss,lambda=' + str(self.config.lambda_weights),
                        self.loss_tracker.result(),
                        step=step)
      tf.summary.scalar('loss_ce,lambda=' + str(self.config.lambda_weights),
                        self.loss_ce_tracker.result(),
                        step=step)
      tf.summary.scalar('loss_weights,lambda=' +
                        str(self.config.lambda_weights),
                        self.loss_mse_tracker.result(),
                        step=step)
      tf.summary.scalar('accuracy,lambda=' + str(self.config.lambda_weights),
                        self.acc_metric.result(),
                        step=step)
    self.loss_tracker.reset_states()
    self.loss_ce_tracker.reset_states()
    self.loss_mse_tracker.reset_states()
    self.acc_metric.reset_states()

  def run(self):
    """ Whole Training pipeline.
            store old weights and compute fisher info before training
        """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print("Current time is " + current_time)
    self.make_dirs()
    self.build_model()
    self.build_tensorboard_writer()
    self.build_loss_and_metric()
    train_ds, val_ds, test_ds, pretrain_ds = self.load_dataset(
        self.config.train_dataset, self.config.train_scene,
        self.config.test_dataset, self.config.test_scene, self.config.data_dir,
        self.config.batch_size)
    self.create_old_params()
    self.create_fisher_params(pretrain_ds)
    self.training(train_ds, val_ds, test_ds)


if __name__ == "__main__":
  experiment = EWC()
  experiment.run()
