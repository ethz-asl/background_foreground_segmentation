import tensorflow as tf
import tensorflow_datasets as tfds
from Nyu_depth_v2_labeled.Nyu_depth_v2_labeled import NyuDepthV2Labeled
import segmentation_models as sm
from tensorflow import keras
import os
import numpy as np
import datetime


@tf.function
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  label = tf.expand_dims(label, axis=2)
  image = tf.cast(image, tf.float32) / 255.
  return image, label


def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask


def load_data(dataset, step, batch_size):
  # load data
  if step == "step1":
    train_ds, train_info = tfds.load(
        dataset,
        split='train[:80%]',
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    val_ds, _ = tfds.load(
        dataset,
        split='train[80%:]',
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )
    test_ds, _ = tfds.load(
        dataset,
        split='test[80%:]',
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )
    lr = 1e-4
  else:
    train_ds, train_info = tfds.load(
        dataset,
        split='test[:80%]',
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    val_ds, _ = tfds.load(
        dataset,
        split='test[80%:]',
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )
    test_ds, _ = tfds.load(
        dataset,
        split='train[80%:]',
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )
    lr = 1e-5

  train_ds = train_ds.map(normalize_img,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_ds = train_ds.cache()
  if step == "step1":
    train_ds = train_ds.shuffle(
        int(train_info.splits['train'].num_examples * 0.8))
  else:
    train_ds = train_ds.shuffle(
        int(train_info.splits['test'].num_examples * 0.8))
  train_ds = train_ds.batch(batch_size)
  train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

  val_ds = val_ds.map(normalize_img,
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  val_ds = val_ds.cache().batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)

  test_ds = test_ds.map(normalize_img,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  test_ds = test_ds.cache().batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)
  return train_ds, val_ds, test_ds, lr


class Model:

  def __init__(self,
               log_dir,
               model_save_dir,
               lr,
               lambda_distillation,
               backbone="vgg16",
               weights=None):
    self.log_dir = log_dir
    self.model_save_dir = model_save_dir
    self.lambda_distillation = lambda_distillation
    self.encoder, self.model = sm.Unet(backbone,
                                       input_shape=(480, 640, 3),
                                       classes=2,
                                       activation='sigmoid',
                                       weights=weights,
                                       encoder_freeze=False)
    self.new_model = keras.Model(
        inputs=self.model.input,
        outputs=[self.encoder.output, self.model.output])
    self.old_encoder, _ = sm.Unet(backbone,
                                  input_shape=(480, 640, 3),
                                  classes=2,
                                  activation='sigmoid',
                                  weights=weights,
                                  encoder_freeze=True)
    self.old_encoder.trainable = False
    self.train_summary_writer = tf.summary.create_file_writer(self.log_dir +
                                                              "/train")
    self.val_summary_writer = tf.summary.create_file_writer(self.log_dir +
                                                            "/val")
    self.test_summary_writer = tf.summary.create_file_writer(self.log_dir +
                                                             "/test")
    self.optimizer = keras.optimizers.Adam(lr)

    self.loss_ce = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    self.loss_mse = keras.losses.MeanSquaredError()
    self.loss_tracker = keras.metrics.Mean('loss', dtype=tf.float32)
    self.loss_ce_tracker = keras.metrics.Mean('loss_ce', dtype=tf.float32)
    self.loss_mse_tracker = keras.metrics.Mean('loss_mse', dtype=tf.float32)
    self.acc_metric = keras.metrics.Accuracy('accuracy')

  def train_step(self, train_x, train_y):
    with tf.GradientTape() as tape:
      [pred_feature, pred_y] = self.new_model(train_x, training=True)
      print(pred_y.shape)
      output_loss = self.loss_ce(train_y, pred_y)
      old_feature = self.old_encoder(train_x, training=False)
      feature_loss = self.loss_mse(old_feature, pred_feature)
      loss = (1 - self.lambda_distillation
             ) * output_loss + self.lambda_distillation * feature_loss
    grads = tape.gradient(loss, self.new_model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.new_model.trainable_weights))
    pred_y = keras.backend.argmax(pred_y, axis=-1)
    self.loss_tracker.update_state(loss)
    self.loss_ce_tracker.update_state(output_loss)
    self.loss_mse_tracker.update_state(feature_loss)
    self.acc_metric.update_state(train_y, pred_y)

  def test_step(self, test_x, test_y):
    [pred_feature, pred_y] = self.new_model(test_x, training=False)
    output_loss = self.loss_ce(test_y, pred_y)
    old_feature = self.old_encoder(test_x, training=False)
    feature_loss = self.loss_mse(old_feature, pred_feature)
    loss = (1 - self.lambda_distillation
           ) * output_loss + self.lambda_distillation * feature_loss
    pred_y = keras.backend.argmax(pred_y, axis=-1)
    # Update val/test metrics
    self.loss_tracker.update_state(loss)
    self.loss_ce_tracker.update_state(output_loss)
    self.loss_mse_tracker.update_state(feature_loss)
    self.acc_metric.update_state(test_y, pred_y)

  def on_epoch_end(self, summary_writer, epoch, mode):
    if mode == "Val" and (epoch + 1) % 10 == 0:
      self.new_model.save(
          os.path.join(
              self.model_save_dir, 'model.' + str(epoch) + '-' +
              str(self.acc_metric.result().numpy())[:5] + '.h5'))
    with summary_writer.as_default():
      tf.summary.scalar('loss,lambda=' + str(self.lambda_distillation),
                        self.loss_tracker.result(),
                        step=epoch)
      tf.summary.scalar('loss_ce,lambda=' + str(self.lambda_distillation),
                        self.loss_ce_tracker.result(),
                        step=epoch)
      tf.summary.scalar('loss_mse,lambda=' + str(self.lambda_distillation),
                        self.loss_mse_tracker.result(),
                        step=epoch)
      tf.summary.scalar('accuracy,lambda=' + str(self.lambda_distillation),
                        self.acc_metric.result(),
                        step=epoch)
    template = ('Epoch {}, ' + mode + ' Loss: {}, ' + mode + ' Loss_ce: {}, ' +
                mode + ' Loss_mse: {}, ' + mode + ' Accuracy: {}')
    print(
        template.format(epoch + 1, self.loss_tracker.result(),
                        self.loss_ce_tracker.result(),
                        self.loss_mse_tracker.result(),
                        self.acc_metric.result() * 100))
    # Reset metrics every epoch
    self.loss_tracker.reset_states()
    self.loss_ce_tracker.reset_states()
    self.loss_mse_tracker.reset_states()
    self.acc_metric.reset_states()


def main():
  # setting parameters
  BACKBONE = "vgg16"
  batch_size = 8
  epochs = 40
  step = "step2"
  # Range: [0,1], lambda=0: no distillation, lambda=1: no train on 2nd task.
  lambda_distillation = 1
  print("lambda_distillation is " + str(lambda_distillation))
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  print("Current time is " + current_time)
  log_dir = os.path.join(
      'exp_EWC', 'lambda' + str(lambda_distillation) + "-epoch" + str(epochs),
      'logs', step)
  model_save_dir = os.path.join(
      'exp_EWC', 'lambda' + str(lambda_distillation) + "-epoch" + str(epochs),
      'saved_model', step)
  if step == "step1":
    saved_weights_dir = None
    lambda_distillation = 0
  else:
    saved_weights_dir = ("exp/lambda0-epoch20/saved_model/step1/"
                         "model.16-0.899.h5")
  try:
    os.makedirs(log_dir + '/train')
    os.makedirs(log_dir + '/val')
    os.makedirs(log_dir + '/test')
    os.makedirs(model_save_dir)
  except os.error:
    pass
  train_ds, val_ds, test_ds, lr = load_data('NyuDepthV2Labeled', step,
                                            batch_size)
  model = Model(log_dir=log_dir,
                model_save_dir=model_save_dir,
                lr=lr,
                lambda_distillation=lambda_distillation,
                backbone=BACKBONE,
                weights=saved_weights_dir)
  #   model.new_model.summary()
  for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    for step, (train_x, train_y) in enumerate(train_ds):
      model.train_step(train_x, train_y)
    model.on_epoch_end(model.train_summary_writer, epoch, mode="Train")
    for (val_x, val_y) in val_ds:
      model.test_step(val_x, val_y)
    model.on_epoch_end(model.val_summary_writer, epoch, mode="Val")
    for (test_x, test_y) in test_ds:
      model.test_step(test_x, test_y)
    model.on_epoch_end(model.test_summary_writer, epoch, mode="Test")


if __name__ == "__main__":
  sm.set_framework('tf.keras')
  main()