import os
import numpy as np
import datetime
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from Nyu_depth_v2_labeled.Nyu_depth_v2_labeled import NyuDepthV2Labeled
import segmentation_models as sm
from tensorflow import keras
from tensorflow.keras import layers
from segmentation_models.segmentation_models.models.unet import Conv3x3BnReLU, DecoderUpsamplingX2Block

# def get_args():
#     parser = argparse.ArgumentParser(description='Progressive Neural Networks')
#     parser.add_argument('-path', default='/local/veniat/data', type=str, help='path to the data')
#     args = parser.parse_known_args()
#     return args[0]


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
    # encoder_freezed = False
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
    pretrain_ds, _ = tfds.load(
        dataset,
        split='train[:80%]',
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )
    lr = 1e-5
    # encoder_freezed = True

  train_ds = train_ds.map(normalize_img,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_ds = train_ds.cache()
  if step == "step1":
    train_ds = train_ds.shuffle(
        int(train_info.splits['train'].num_examples * 0.8))
  else:
    train_ds = train_ds.shuffle(
        int(train_info.splits['test'].num_examples * 0.8))
  train_ds = train_ds.batch(batch_size)  #.repeat()
  train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

  val_ds = val_ds.map(normalize_img,
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  val_ds = val_ds.cache().batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)

  test_ds = test_ds.map(normalize_img,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  test_ds = test_ds.cache().batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)
  pretrain_ds = pretrain_ds.map(
      normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  pretrain_ds = pretrain_ds.cache().batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)
  return train_ds, val_ds, test_ds, pretrain_ds, lr


class myMultiply(layers.Layer):

  def __init__(self, num_outputs, curr_block_name, **kwargs):
    super(myMultiply, self).__init__(name=curr_block_name + "_mul", **kwargs)
    self.num_outputs = num_outputs
    # self._name = curr_block_name+"_mul"
    self.curr_block_name = curr_block_name

  def build(self, input_shape):  # Create the state of the layer (weights)
    alpha_init = tf.random.uniform([1, 1, 1, self.num_outputs],
                                   maxval=1 / self.num_outputs)
    shape = tf.constant([batch_size, input_shape[1], input_shape[2], 1],
                        tf.int32)
    alpha_init = tf.tile(alpha_init, shape)
    self.alpha = tf.Variable(alpha_init,
                             trainable=True,
                             name=self.curr_block_name + "_alpha")

  def call(self, inputs):  # Defines the computation from inputs to outputs
    output = layers.Multiply()([tf.convert_to_tensor(self.alpha), inputs])
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


def build_model_with_adaptor(backbone, weights):
  # Now I hard code using backbone "vgg16" and "Unet"!!!
  _, new_model = sm.Unet(backbone,
                         input_shape=(480, 640, 3),
                         classes=2,
                         activation='sigmoid',
                         weights=None,
                         encoder_freeze=False)
  _, old_model = sm.Unet(backbone,
                         input_shape=(480, 640, 3),
                         classes=2,
                         activation='sigmoid',
                         weights=weights,
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
  model = keras.Model(inputs=[new_input, old_input], outputs=new_x)
  return model


def build_model_wo_adaptor(backbone, weights):
  encoder, model = sm.Unet(backbone,
                           input_shape=(480, 640, 3),
                           classes=2,
                           activation='sigmoid',
                           weights=weights,
                           encoder_freeze=False)
  new_model = keras.Model(inputs=model.input,
                          outputs=[encoder.output, model.output])
  return new_model


class Model:

  def __init__(self,
               log_dir,
               model_save_dir,
               lr,
               lambda_weights,
               progress_method,
               backbone="vgg16",
               new_weights=None,
               old_weights=None):
    self.log_dir = log_dir
    self.model_save_dir = model_save_dir
    self.lambda_weights = lambda_weights
    self.progress_method = progress_method
    if self.progress_method == "fine_tune":
      self.new_model = build_model_wo_adaptor(backbone, new_weights)
    else:
      self.new_model = build_model_with_adaptor(backbone, None)
      # self.new_model.summary()
      self.new_model.load_weights(new_weights)
    self.new_model.trainable = False
    self.old_model = build_model_wo_adaptor(backbone, old_weights)
    self.train_summary_writer = tf.summary.create_file_writer(self.log_dir +
                                                              "/train")
    self.val_summary_writer = tf.summary.create_file_writer(self.log_dir +
                                                            "/val")
    self.test_summary_writer = tf.summary.create_file_writer(self.log_dir +
                                                             "/test")
    self.optimizer = keras.optimizers.Adam(lr)
    self.loss_kl = keras.losses.KLDivergence()
    self.loss_mse = keras.losses.MeanSquaredError()
    self.loss_tracker = keras.metrics.Mean('loss', dtype=tf.float32)
    self.loss_kl_tracker = keras.metrics.Mean('loss_kl', dtype=tf.float32)
    self.loss_mse_tracker = keras.metrics.Mean('loss_mse', dtype=tf.float32)
    self.acc_metric = keras.metrics.Accuracy('accuracy')

  def create_old_params(self):
    self.old_params = []
    for param in self.old_model.trainable_weights:
      old_param_name = param.name.replace(':0', '_old')
      self.old_params.append(
          tf.Variable(param, trainable=False, name=old_param_name))

  def create_fisher_params(self, dataset):
    self.fisher_params = []
    grads_list = [
    ]  # list of list of gradients, outer: for different batches, inner: for different network parameters
    for step, (x, y) in enumerate(dataset):
      if step > 40:
        break
      with tf.GradientTape() as tape:
        [_, pred_y] = self.old_model(x, training=True)
        log_y = tf.math.log(pred_y)
        y = tf.cast(y, log_y.dtype)
        log_likelihood = tf.reduce_sum(y * log_y[:, :, :, 1:2] +
                                       (1 - y) * log_y[:, :, :, 0:1],
                                       axis=[1, 2, 3])
      grads = tape.gradient(log_likelihood, self.old_model.trainable_weights)
      grads_list.append(grads)
    fisher_params = []
    fisher_param_names = [
        param.name.replace(':0', '_fisher')
        for param in self.old_model.trainable_weights
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
    losses = []
    for i, param in enumerate(self.old_model.trainable_weights):
      losses.append(
          tf.reduce_sum(self.fisher_params[i] *
                        (param - self.old_params[i])**2))
    return tf.reduce_sum(losses)

  def train_step(self, train_x, train_y):
    # print(self.new_model.trainable_weights[0])
    with tf.GradientTape() as tape:
      if self.progress_method == "fine_tune":
        [_, pred_new_y] = self.new_model(train_x, training=False)
      else:
        pred_new_y = self.new_model([train_x, train_x], training=False)
      [_, pred_old_y] = self.old_model(train_x, training=True)
      output_loss = self.loss_kl(pred_new_y, pred_old_y)
      loss = (
          1 - self.lambda_weights
      ) * output_loss + self.lambda_weights * self.compute_consolidation_loss()
    grads = tape.gradient(loss, self.old_model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.old_model.trainable_weights))
    pred_y = tf.math.argmax(pred_old_y, axis=-1)
    self.loss_tracker.update_state(loss)
    self.loss_kl_tracker.update_state(output_loss)
    self.loss_mse_tracker.update_state(self.compute_consolidation_loss())
    self.acc_metric.update_state(train_y, pred_y)

  def test_step(self, test_x, test_y):
    if self.progress_method == "fine_tune":
      [_, pred_new_y] = self.new_model(test_x, training=False)
    else:
      pred_new_y = self.new_model([test_x, test_x], training=False)
    [_, pred_old_y] = self.old_model(test_x, training=False)
    output_loss = self.loss_kl(pred_new_y, pred_old_y)
    loss = (
        1 - self.lambda_weights
    ) * output_loss + self.lambda_weights * self.compute_consolidation_loss()
    pred_y = tf.math.argmax(pred_old_y, axis=-1)
    # Update val/test metrics
    self.loss_tracker.update_state(loss)
    self.loss_kl_tracker.update_state(output_loss)
    self.loss_mse_tracker.update_state(self.compute_consolidation_loss())
    self.acc_metric.update_state(test_y, pred_y)

  def on_epoch_end(self, summary_writer, epoch, mode):
    # if mode == "Val" and (epoch + 1) % 10 == 0:
    #     self.new_model.save(
    #         os.path.join(
    #             self.model_save_dir, 'model.' + str(epoch) + '-' +
    #             str(self.acc_metric.result().numpy())[:5] + '.h5'))
    with summary_writer.as_default():
      tf.summary.scalar('loss,lambda=' + str(self.lambda_weights),
                        self.loss_tracker.result(),
                        step=epoch)
      tf.summary.scalar('loss_kl,lambda=' + str(self.lambda_weights),
                        self.loss_kl_tracker.result(),
                        step=epoch)
      tf.summary.scalar('loss_mse,lambda=' + str(self.lambda_weights),
                        self.loss_mse_tracker.result(),
                        step=epoch)
      tf.summary.scalar('accuracy,lambda=' + str(self.lambda_weights),
                        self.acc_metric.result(),
                        step=epoch)
    template = 'Epoch {}, ' + mode + ' Loss: {}, ' + mode + ' Loss_kl: {}, ' + mode + ' Loss_mse: {}, ' + mode + ' Accuracy: {}'
    print(
        template.format(epoch + 1, self.loss_tracker.result(),
                        self.loss_kl_tracker.result(),
                        self.loss_mse_tracker.result(),
                        self.acc_metric.result() * 100))
    # Reset metrics every epoch
    self.loss_tracker.reset_states()
    self.loss_kl_tracker.reset_states()
    self.loss_mse_tracker.reset_states()
    self.acc_metric.reset_states()


def main():
  # setting parameters
  BACKBONE = "vgg16"
  global batch_size
  batch_size = 1
  epochs = 40
  step = "step2"
  progress_method = "lateral"  # fine_tune / lateral
  lambda_weights = 0.1  # range: [0,1], lambda=0: no weight constraints, lambda=1: no train on 2nd task
  print("lambda_weights is " + str(lambda_weights))
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  print("Current time is " + current_time)
  log_dir = os.path.join('exp_PC', 'Compress', progress_method,
                         'lambda' + str(lambda_weights), 'logs')
  model_save_dir = os.path.join('exp_PC', 'Compress', progress_method,
                                'lambda' + str(lambda_weights), 'saved_model')
  old_weights_dir = "exp/lambda0-epoch20/saved_model/step1/model.16-0.899.h5"
  if progress_method == "fine_tune":
    new_weights_dir = "exp_PC/Progress/finetune/lambda0-epoch40/saved_model/step2/model.19-0.921.h5"
  else:
    new_weights_dir = "exp_PC/Progress/lateral/saved_model/step2/model.9-0.910.h5"
  try:
    os.makedirs(log_dir + '/train')
    os.makedirs(log_dir + '/val')
    os.makedirs(log_dir + '/test')
    os.makedirs(model_save_dir)
  except os.error:
    pass
  train_ds, val_ds, test_ds, pretrain_ds, lr = load_data(
      'NyuDepthV2Labeled', step, batch_size)
  model = Model(log_dir=log_dir,
                model_save_dir=model_save_dir,
                lr=lr,
                lambda_weights=lambda_weights,
                progress_method=progress_method,
                backbone=BACKBONE,
                new_weights=new_weights_dir,
                old_weights=old_weights_dir)
  model.create_old_params()
  model.create_fisher_params(pretrain_ds)
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

  # i = 0
  # for image, _ in train_ds:
  #   features_maps = encoder(image)
  #   features_maps = np.array(features_maps)
  #   np.save("old_features/batch_"+str('{:03d}'.format(i)+".npy"),features_maps)
  #   print("Predicting train batch %d" % i)
  #   i = i + 1

  # testing_save_path = './test_result_epoch10_diffscene/'
  # validating_save_path = './val_result_epoch10_diffscene/'
  # if os.path.exists(testing_save_path) == False:
  #   os.mkdir(testing_save_path)
  # if os.path.exists(validating_save_path) == False:
  #   os.mkdir(validating_save_path)
  # i = 0
  # for image, label in test_ds:
  #   for j in range(image.shape[0]):
  #     pred_label = model.predict(image)
  #     Image.save_img(
  #         os.path.join(testing_save_path,
  #                      str(i * batch_size + j).zfill(4) + '_image.png'),
  #         image[j])
  #     Image.save_img(
  #         os.path.join(testing_save_path,
  #                      str(i * batch_size + j).zfill(4) + '_trueseg.png'),
  #         label[j])
  #     # print(label[j])
  #     Image.save_img(
  #         os.path.join(testing_save_path,
  #                      str(i * batch_size + j).zfill(4) + '_preddeg.png'),
  #         create_mask(pred_label[j]))
  #     # print(create_mask(pred_label[j]))
  #   print("Predicting test batch %d" % i)
  #   i = i + 1
  # i = 0
  # for image, label in val_ds:
  #   for j in range(image.shape[0]):
  #     pred_label = model.predict(image)
  #     Image.save_img(
  #         os.path.join(validating_save_path,
  #                      str(i * batch_size + j).zfill(4) + '_image.png'),
  #         image[j])
  #     Image.save_img(
  #         os.path.join(validating_save_path,
  #                      str(i * batch_size + j).zfill(4) + '_trueseg.png'),
  #         label[j])
  #     Image.save_img(
  #         os.path.join(validating_save_path,
  #                      str(i * batch_size + j).zfill(4) + '_preddeg.png'),
  #         create_mask(pred_label[j]))
  #   print("Predicting train batch %d" % i)
  #   i = i + 1


if __name__ == "__main__":
  sm.set_framework('tf.keras')
  main()
