import sys
sys.path.append("..")
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import argparse
import numpy as np
import datetime
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import bfseg.data.nyu.Nyu_depth_v2_labeled
import bfseg.data.meshdist.bfseg_cla_meshdist_labels
import segmentation_models as sm
import tensorflow.keras.preprocessing.image as Image
from train_binary_segmodel_EWC import EWC


class Compress(EWC):
    """
    Experiment to train on 2nd task: Step2->compress
    "Progress & Compress: A scalable framework for continual learning (P&C)"
    https://arxiv.org/pdf/1805.06370.pdf
    """
    def __init__(self):
        super(Compress, self).__init__()
        global batch_size
        batch_size = self.config.batch_size
        self.log_dir = os.path.join('../experiments', self.config.exp_name,
                                    'lambda' + str(self.config.lambda_weights),
                                    'logs')
        self.model_save_dir = os.path.join(
            '../experiments', self.config.exp_name,
            'lambda' + str(self.config.lambda_weights), 'saved_model')

    def _addArguments(self, parser):
        """ Add custom arguments that are needed for this experiment """
        super(Compress, self)._addArguments(parser)
        parser.add_argument(
            '-type_progress',
            default="lateral_connection",
            type=str,
            help=
            'method used in progress stage: 1.fine_tune, 2.lateral_connection')
        parser.add_argument(
            '-pretrained_dir2',
            default=None,
            type=str,
            help='directory of pretrained model weights on 2nd task')

    def build_model(self):
        """ Build both new model(train) and old model(guide/constraint)
            new_model is the knowledge base (model trained on 1st task), 
            old model is the active column (model trained on 2nd task)
        """
        if self.config.type_progress == "lateral_connection":
            self.old_model = keras.models.load_model(
                self.config.pretrained_dir2)
        elif self.config.type_progress == "fine_tune":
            _, self.old_model = sm.Unet(self.config.backbone,
                                        input_shape=(self.config.image_h,
                                                     self.config.image_w, 3),
                                        classes=2,
                                        activation='sigmoid',
                                        weights=self.config.pretrained_dir2,
                                        encoder_freeze=True)
        self.old_model.trainable = False
        encoder, model = sm.Unet(self.config.backbone,
                                 input_shape=(self.config.image_h,
                                              self.config.image_w, 3),
                                 classes=2,
                                 activation='sigmoid',
                                 weights=self.config.pretrained_dir,
                                 encoder_freeze=False)
        self.new_model = keras.Model(inputs=model.input,
                                     outputs=[encoder.output, model.output])

    def build_loss_and_metric(self):
        """ Add loss criteria and metrics"""
        self.loss_kl = keras.losses.KLDivergence()
        self.loss_mse = keras.losses.MeanSquaredError()
        self.loss_tracker = keras.metrics.Mean('loss', dtype=tf.float32)
        self.loss_kl_tracker = keras.metrics.Mean('loss_kl', dtype=tf.float32)
        self.loss_mse_tracker = keras.metrics.Mean('loss_weights',
                                                   dtype=tf.float32)
        self.acc_metric = keras.metrics.Accuracy('accuracy')

    def train_step(self, train_x, train_y, train_m, step):
        """ Training on one batch:
            Compute masked Kl divergence loss(true label, predicted label)
            + Weight consolidation loss(new weights, old weights)
            update losses & metrics
        """
        with tf.GradientTape() as tape:
            if self.config.type_progress == "fine_tune":
                pred_old_y = self.old_model(train_x, training=False)
            elif self.config.type_progress == "lateral_connection":
                pred_old_y = self.old_model([train_x, train_x], training=False)
            [_, pred_new_y] = self.new_model(train_x, training=True)
            pred_old_y_masked = tf.boolean_mask(pred_old_y, train_m)
            pred_new_y_masked = tf.boolean_mask(pred_new_y, train_m)
            output_loss = self.loss_kl(pred_old_y_masked, pred_new_y_masked)
            consolidation_loss = self.compute_consolidation_loss()
            loss = (
                1 - self.config.lambda_weights
            ) * output_loss + self.config.lambda_weights * consolidation_loss
        grads = tape.gradient(loss, self.new_model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.new_model.trainable_weights))
        pred_new_y = tf.math.argmax(pred_new_y, axis=-1)
        pred_new_y_masked = tf.boolean_mask(pred_new_y, train_m)
        train_y_masked = tf.boolean_mask(train_y, train_m)
        self.acc_metric.update_state(train_y_masked, pred_new_y_masked)

        if self.config.tensorboard_write_freq == "batch":
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss,lambda=' +
                                  str(self.config.lambda_weights),
                                  loss,
                                  step=step)
                tf.summary.scalar('loss_kl,lambda=' +
                                  str(self.config.lambda_weights),
                                  output_loss,
                                  step=step)
                tf.summary.scalar('loss_weights,lambda=' +
                                  str(self.config.lambda_weights),
                                  consolidation_loss,
                                  step=step)
                tf.summary.scalar('accuracy,lambda=' +
                                  str(self.config.lambda_weights),
                                  self.acc_metric.result(),
                                  step=step)
                self.acc_metric.reset_states()
        elif self.config.tensorboard_write_freq == "epoch":
            self.loss_tracker.update_state(loss)
            self.loss_kl_tracker.update_state(output_loss)
            self.loss_mse_tracker.update_state(consolidation_loss)
        else:
            raise Exception("Invalid tensorboard_write_freq: %s!" %
                            self.config.tensorboard_write_freq)

    def test_step(self, test_x, test_y, test_m):
        """ Validating/Testing on one batch
            update losses & metrics
        """
        if self.config.type_progress == "fine_tune":
            pred_old_y = self.old_model(test_x, training=False)
        elif self.config.type_progress == "lateral_connection":
            pred_old_y = self.old_model([test_x, test_x], training=False)
        [_, pred_new_y] = self.new_model(test_x, training=False)
        pred_old_y_masked = tf.boolean_mask(pred_old_y, test_m)
        pred_new_y_masked = tf.boolean_mask(pred_new_y, test_m)
        output_loss = self.loss_kl(pred_old_y_masked, pred_new_y_masked)
        consolidation_loss = self.compute_consolidation_loss()
        loss = (
            1 - self.config.lambda_weights
        ) * output_loss + self.config.lambda_weights * consolidation_loss
        pred_new_y = tf.math.argmax(pred_new_y, axis=-1)
        pred_new_y_masked = tf.boolean_mask(pred_new_y, test_m)
        test_y_masked = tf.boolean_mask(test_y, test_m)
        self.loss_tracker.update_state(loss)
        self.loss_kl_tracker.update_state(output_loss)
        self.loss_mse_tracker.update_state(consolidation_loss)
        self.acc_metric.update_state(test_y_masked, pred_new_y_masked)

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
            self.loss_kl_tracker.reset_states()
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
            tf.summary.scalar('loss_kl,lambda=' +
                              str(self.config.lambda_weights),
                              self.loss_kl_tracker.result(),
                              step=step)
            tf.summary.scalar('loss_weights,lambda=' +
                              str(self.config.lambda_weights),
                              self.loss_mse_tracker.result(),
                              step=step)
            tf.summary.scalar('accuracy,lambda=' +
                              str(self.config.lambda_weights),
                              self.acc_metric.result(),
                              step=step)
        self.loss_tracker.reset_states()
        self.loss_kl_tracker.reset_states()
        self.loss_mse_tracker.reset_states()
        self.acc_metric.reset_states()

    def training(self, train_ds, val_ds, test_ds):
        """
        train for assigned epochs, and validate & test after each epoch/batch,
        save models after every several epochs
        """
        step = 0
        for epoch in range(self.config.epochs):
            print("\nStart of epoch %d" % (epoch, ))
            if self.config.tensorboard_write_freq == "batch":
                for train_x, train_y, train_m in train_ds:
                    if train_x.shape[0] == self.config.batch_size:
                        self.train_step(train_x, train_y, train_m, step)
                        for val_x, val_y, val_m in val_ds:
                            if val_x.shape[0] == self.config.batch_size:
                                self.test_step(val_x, val_y, val_m)
                        self.write_to_tensorboard(self.val_summary_writer,
                                                  step)
                        for test_x, test_y, test_m in test_ds:
                            if test_x.shape[0] == self.config.batch_size:
                                self.test_step(test_x, test_y, test_m)
                        self.write_to_tensorboard(self.test_summary_writer,
                                                  step)
                        step += 1
                if epoch == 0:
                    print("There are %d batches in the training dataset" %
                          step)
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
    experiment = Compress()
    experiment.run()
