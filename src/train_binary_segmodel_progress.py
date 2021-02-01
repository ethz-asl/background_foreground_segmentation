import sys
sys.path.append("..")
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import segmentation_models as sm
from bfseg.cl_experiments import BaseSegExperiment


class myMultiply(layers.Layer):

  def __init__(self, num_outputs, curr_block_name, **kwargs):
    super(myMultiply, self).__init__(name=curr_block_name + "_mul", **kwargs)
    self.num_outputs = num_outputs
    self.curr_block_name = curr_block_name

  def build(self, input_shape):  # Create the state of the layer (weights)
    alpha_init = tf.random.uniform([1, 1, 1, self.num_outputs],
                                   maxval=1 / self.num_outputs)
    self.shape = tf.constant([batch_size, input_shape[1], input_shape[2], 1],
                             tf.int32)
    self.alpha = tf.Variable(alpha_init,
                             trainable=True,
                             name=self.curr_block_name + "_alpha")

  def call(self, inputs):  # Defines the computation from inputs to outputs
    alpha = tf.tile(self.alpha, self.shape)
    output = layers.Multiply()([alpha, inputs])
    return output

  def get_config(self):
    config = {
        "num_outputs": self.num_outputs,
        "curr_block_name": self.curr_block_name
    }
    base_config = super(myMultiply, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def add_lateral_connect(new_layer, old_layer, new_input, old_input,
                        curr_block_name):
  filter_num = new_layer.output_shape[-1]
  new_x_1 = new_layer(new_input)
  old_output = old_layer(old_input)
  new_x_2 = layers.ReLU(name=curr_block_name + "_relu_connect")(old_output)
  # new_x_2 = layers.Conv2D(filter_num,1,name=curr_block_name+"_conv_connect")(new_x_2)
  new_x_2 = layers.Conv2D(filter_num,
                          1,
                          kernel_initializer="ones",
                          name=curr_block_name + "_conv_connect")(new_x_2)
  multiply_layer = myMultiply(filter_num, curr_block_name)
  new_x_2 = multiply_layer(new_x_2)
  new_output = layers.Add(name=curr_block_name + "_add")([new_x_1, new_x_2])
  return new_output, old_output


class Progress(BaseSegExperiment):
  """
    Experiment to train on 2nd task with lateral connection: Step1->progress
    "Progress & Compress: A scalable framework for continual learning (P&C)"
    https://arxiv.org/pdf/1805.06370.pdf
    """

  def __init__(self):
    super(Progress, self).__init__()
    global batch_size
    batch_size = self.config.batch_size

  def build_model(self):
    """ Build both new model(train) and old model(guide/constraint)
            Now I hard code using backbone "vgg16" and "Unet"
        """
    _, new_model = sm.Unet(self.config.backbone,
                           input_shape=(self.config.image_h,
                                        self.config.image_w, 3),
                           classes=2,
                           activation='sigmoid',
                           weights=None,
                           encoder_freeze=False)
    _, old_model = sm.Unet(self.config.backbone,
                           input_shape=(self.config.image_h,
                                        self.config.image_w, 3),
                           classes=2,
                           activation='sigmoid',
                           weights=self.config.pretrained_dir,
                           encoder_freeze=True)
    old_model.trainable = False
    for layer in old_model.layers:
      layer._name = layer.name + str("_old")
    new_input = new_model.input
    old_input = old_model.input

    new_skip_connection_layers = [
        'block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2'
    ]
    old_skip_connection_layers = [
        'block5_conv3_old', 'block4_conv3_old', 'block3_conv3_old',
        'block2_conv2_old'
    ]
    new_skips = [
        new_model.get_layer(name=i).output for i in new_skip_connection_layers
    ]
    old_skips = [
        old_model.get_layer(name=i).output for i in old_skip_connection_layers
    ]
    for i in range(18):
      if i == 0:
        new_x = new_input
        old_x = old_input
        continue
      pre_layer_name = new_model.layers[i - 1].name
      new_layer = new_model.layers[i]
      old_layer = old_model.layers[i]
      if pre_layer_name.endswith("pool"):
        curr_block_name = new_layer.name[:6]
        new_x, old_x = add_lateral_connect(new_layer, old_layer, new_x, old_x,
                                           curr_block_name)
        new_x = layers.ReLU(name=curr_block_name + "_relu")(new_x)
      else:
        new_x = new_layer(new_x)
        old_x = old_layer(old_x)
    skip_id = 0
    for i in range(18, len(new_model.layers)):
      new_layer = new_model.layers[i]
      old_layer = old_model.layers[i]
      curr_layer_name = new_layer.name
      if curr_layer_name.startswith("decoder") and curr_layer_name.endswith(
          "conv"):
        curr_block_name = curr_layer_name[:-5]
        new_x, old_x = add_lateral_connect(new_layer, old_layer, new_x, old_x,
                                           curr_block_name)
      elif curr_layer_name.endswith("concat"):
        new_x = new_layer([new_x, new_skips[skip_id]])
        old_x = old_layer([old_x, old_skips[skip_id]])
        skip_id += 1
      else:
        new_x = new_layer(new_x)
        old_x = old_layer(old_x)
    self.new_model = keras.Model(inputs=[new_input, old_input], outputs=new_x)

  def train_step(self, train_x, train_y, train_m, step):
    """ Training on one batch:
            Compute masked cross entropy loss(true label, predicted label),
            update losses & metrics
        """
    with tf.GradientTape() as tape:
      pred_y = self.new_model([train_x, train_x], training=True)
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
    pred_y = self.new_model([test_x, test_x], training=False)
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
        if val_x.shape[0] == self.config.batch_size:
          self.test_step(val_x, val_y, val_m)
      self.new_model.save(
          os.path.join(
              self.model_save_dir, 'model.' + str(epoch) + '-' +
              str(self.acc_metric.result().numpy())[:5] + '.tf'))
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
          if train_x.shape[0] == self.config.batch_size:
            self.train_step(train_x, train_y, train_m, step)
            for val_x, val_y, val_m in val_ds:
              if val_x.shape[0] == self.config.batch_size:
                self.test_step(val_x, val_y, val_m)
            self.write_to_tensorboard(self.val_summary_writer, step)
            for test_x, test_y, test_m in test_ds:
              if test_x.shape[0] == self.config.batch_size:
                self.test_step(test_x, test_y, test_m)
            self.write_to_tensorboard(self.test_summary_writer, step)
            step += 1
        if epoch == 0:
          print("There are %d batches in the training dataset" % step)
      elif self.config.tensorboard_write_freq == "epoch":
        for train_x, train_y, train_m in train_ds:
          if train_x.shape[0] == self.config.batch_size:
            self.train_step(train_x, train_y, train_m, step)
        self.write_to_tensorboard(self.train_summary_writer, epoch)
        for val_x, val_y, val_m in val_ds:
          if val_x.shape[0] == self.config.batch_size:
            self.test_step(val_x, val_y, val_m)
        self.write_to_tensorboard(self.val_summary_writer, epoch)
        for test_x, test_y, test_m in test_ds:
          if test_x.shape[0] == self.config.batch_size:
            self.test_step(test_x, test_y, test_m)
        self.write_to_tensorboard(self.test_summary_writer, epoch)
      self.on_epoch_end(epoch, val_ds)


if __name__ == "__main__":
  experiment = Progress()
  experiment.run()
