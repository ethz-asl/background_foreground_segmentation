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
            shuffle_files=True,
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
            shuffle_files=True,
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
        self.train_summary_writer = tf.summary.create_file_writer(
            self.log_dir + "/train")
        self.val_summary_writer = tf.summary.create_file_writer(self.log_dir +
                                                                "/val")
        self.test_summary_writer = tf.summary.create_file_writer(self.log_dir +
                                                                 "/test")
        self.optimizer = keras.optimizers.Adam(lr)

        self.loss_ce = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False)
        self.loss_mse = keras.losses.MeanSquaredError()
        self.loss_tracker = keras.metrics.Mean('loss', dtype=tf.float32)
        self.loss_ce_tracker = keras.metrics.Mean('loss_ce', dtype=tf.float32)
        self.loss_mse_tracker = keras.metrics.Mean('loss_mse',
                                                   dtype=tf.float32)
        self.acc_metric = keras.metrics.Accuracy('accuracy')

    def create_old_params(self):
        self.old_params = []
        for param in self.new_model.trainable_weights:
            old_param_name = param.name.replace(':0', '_old')
            self.old_params.append(tf.Variable(param.value,trainable=False,name=old_param_name))

    def create_fisher_params(self, dataset, num_batch):
        self.fisher_params = []
        log_liklihoods = []
        for step, (x, y) in enumerate(dataset):
            if step > num_batch:
                break
            with tf.GradientTape() as tape:
                [_, pred_y] = self.new_model(x, training=True)
                log_y = tf.log(pred_y)
                y = tf.cast(y,log_y.dtype)
                log_liklihood = tf.reduce_mean(y*log_y[:,:,:,1]+(1-y)*log_y[:,:,:,0],axis=[1,2])
                log_liklihoods.append(log_liklihood)
            log_likelihood = tf.reduce_mean(tf.concat(log_liklihoods))
            grads = tape.gradient(log_likelihood, self.new_model.trainable_weights)
            fisher_param_names = [param.name.replace(':0', '_fisher') for param in self.new_model.trainable_weights]
            for param_name, param in zip(fisher_param_names, grads):
                self.fisher_params.append(tf.Variable(tf.square(param).value,trainable=False,name=param_name))

    def compute_consolidation_loss(self):
        try:
            losses = []
            for i, param in enumerate(self.new_model.trainable_weights):
                losses.append(tf.reduce_sum(self.fisher_params[i] * (param - self.old_params[i])**2)))
            return tf.reduce_sum(losses)
        except AttributeError:
            return 0

    def train_step(self, train_x, train_y):
        with tf.GradientTape() as tape:
            [pred_feature, pred_y] = self.new_model(train_x, training=True)
            output_loss = self.loss_ce(train_y, pred_y)
            old_feature = self.old_encoder(train_x, training=False)
            feature_loss = self.loss_mse(old_feature, pred_feature)
            loss = (1 - self.lambda_distillation
                    ) * output_loss + self.lambda_distillation * feature_loss
        grads = tape.gradient(loss, self.new_model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.new_model.trainable_weights))
        pred_y = tf.math.argmax(pred_y, axis=-1)
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
        pred_y = tf.math.argmax(pred_y, axis=-1)
        # Update val/test metrics
        self.loss_tracker.update_state(loss)
        self.loss_ce_tracker.update_state(output_loss)
        self.loss_mse_tracker.update_state(feature_loss)
        self.acc_metric.update_state(test_y, pred_y)

    def on_epoch_end(self, summary_writer, epoch, mode):
        if mode == "Val":
            self.model.save(
                os.path.join(
                    self.model_save_dir, 'model.' + str(epoch) + '-' +
                    str(self.acc_metric.result().numpy())[:5] + '.h5'))
        with summary_writer.as_default():
            tf.summary.scalar('loss,lambda=' + str(self.lambda_distillation),
                              self.loss_tracker.result(),
                              step=epoch)
            tf.summary.scalar('loss_ce,lambda=' +
                              str(self.lambda_distillation),
                              self.loss_ce_tracker.result(),
                              step=epoch)
            tf.summary.scalar('loss_mse,lambda=' +
                              str(self.lambda_distillation),
                              self.loss_mse_tracker.result(),
                              step=epoch)
            tf.summary.scalar('accuracy,lambda=' +
                              str(self.lambda_distillation),
                              self.acc_metric.result(),
                              step=epoch)
        template = 'Epoch {}, ' + mode + ' Loss: {}, ' + mode + ' Loss_ce: {}, ' + mode + ' Loss_mse: {}, ' + mode + ' Accuracy: {}'
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
    lambda_distillation = 1  # range: [0,1], lambda=0: no distillation, lambda=1: no train on 2nd task
    print("lambda_distillation is " + str(lambda_distillation))
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print("Current time is "+current_time)
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
        saved_weights_dir = "exp/lambda0-epoch20/saved_model/step1/model.16-0.899.h5"
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
    model.create_old_params()
    model.create_fisher_params(test_ds, num_batch=100)
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch, ))
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
