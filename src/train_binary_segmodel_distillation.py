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
import bfseg.data.nyu.Nyu_depth_v2_labeled
import bfseg.data.meshdist.bfseg_cla_meshdist_labels
import segmentation_models as sm
import tensorflow.keras.preprocessing.image as Image
from train_binary_segmodel_base import Base


class Distillation(Base):
    """
    Experiment to train on 2nd task with extra distillation loss:
    Distillation loss: 1. feature -> Distillation on Intermediate Feature Space (Encoder output)
                       2. output  -> Distillation on the Output Layer (~LWF)
    """
    def __init__(self):
        super(Distillation, self).__init__()
        print("lambda_distillation is " + str(self.config.lambda_distillation))
        self.log_dir = os.path.join(
            '../experiments', self.config.exp_name,
            'lambda' + str(self.config.lambda_distillation), 'logs')
        self.model_save_dir = os.path.join(
            '../experiments', self.config.exp_name,
            'lambda' + str(self.config.lambda_distillation), 'saved_model')

    def _addArguments(self, parser):
        """ Add custom arguments that are needed for this experiment """
        super(Distillation, self)._addArguments(parser)
        parser.add_argument(
            '-lambda_distillation',
            default=0,
            type=float,
            help=
            'weighting constant, range: [0,1], lambda=0: no distillation, lambda=1: no train on 2nd task'
        )
        parser.add_argument('-type_distillation',
                            default="feature",
                            type=str,
                            help='distillation on feature space/output layer')

    def build_model(self):
        """ Build both new model(train) and old model(guide/constraint)"""
        self.encoder, self.model = sm.Unet(self.config.backbone,
                                           input_shape=(self.config.image_h,
                                                        self.config.image_w,
                                                        3),
                                           classes=2,
                                           activation='sigmoid',
                                           weights=self.config.pretrained_dir,
                                           encoder_freeze=False)
        self.new_model = keras.Model(
            inputs=self.model.input,
            outputs=[self.encoder.output, self.model.output])
        if self.config.type_distillation == "feature":
            self.old_encoder, _ = sm.Unet(self.config.backbone,
                                          input_shape=(self.config.image_h,
                                                       self.config.image_w, 3),
                                          classes=2,
                                          activation='sigmoid',
                                          weights=self.config.pretrained_dir,
                                          encoder_freeze=True)
            self.old_encoder.trainable = False
        elif self.config.type_distillation == "output":
            _, self.old_model = sm.Unet(self.config.backbone,
                                        input_shape=(self.config.image_h,
                                                     self.config.image_w, 3),
                                        classes=2,
                                        activation='sigmoid',
                                        weights=self.config.pretrained_dir,
                                        encoder_freeze=True)
            self.old_model.trainable = False
        else:
            raise Exception(
                "Distillation type %s not found, pick one from 'feature/output'!"
                % self.config.type_distillation)

    def build_loss_and_metric(self):
        """ Add loss criteria and metrics"""
        self.loss_ce = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False)
        self.loss_mse = keras.losses.MeanSquaredError()
        self.loss_tracker = keras.metrics.Mean('loss', dtype=tf.float32)
        self.loss_ce_tracker = keras.metrics.Mean('loss_ce', dtype=tf.float32)
        self.loss_distillation_tracker = keras.metrics.Mean('loss_distill',
                                                            dtype=tf.float32)
        self.acc_metric = keras.metrics.Accuracy('accuracy')

    def train_step(self, train_x, train_y, train_m, step):
        """ Training on one batch:
            Compute masked cross entropy loss(true label, predicted label)
            + Distillation type: 1. feature -> MSE(new_encoder output, old_encoder output)
                                 2. output  -> masked cross entropy loss(pseudo label, predicted label),
            update losses & metrics
        """
        with tf.GradientTape() as tape:
            train_y_masked = tf.boolean_mask(train_y, train_m)
            if self.config.type_distillation == "feature":
                [pred_feature, pred_y] = self.new_model(train_x, training=True)
                pred_y_masked = tf.boolean_mask(pred_y, train_m)
                old_feature = self.old_encoder(train_x, training=False)
                distillation_loss = self.loss_mse(
                    tf.stop_gradient(old_feature), pred_feature)
            elif self.config.type_distillation == "output":
                [_, pred_y] = self.new_model(train_x, training=True)
                pred_y_masked = tf.boolean_mask(pred_y, train_m)
                pseudo_y = self.old_model(train_x, training=False)
                pseudo_y = keras.backend.argmax(pseudo_y, axis=-1)
                pseudo_y_masked = tf.boolean_mask(pseudo_y, train_m)
                distillation_loss = self.loss_ce(
                    tf.stop_gradient(pseudo_y_masked), pred_y_masked)
            output_loss = self.loss_ce(train_y_masked, pred_y_masked)
            loss = (
                1 - self.config.lambda_distillation
            ) * output_loss + self.config.lambda_distillation * distillation_loss
        grads = tape.gradient(loss, self.new_model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.new_model.trainable_weights))
        pred_y = tf.math.argmax(pred_y, axis=-1)
        pred_y_masked = tf.boolean_mask(pred_y, train_m)
        self.acc_metric.update_state(train_y_masked, pred_y_masked)

        if self.config.tensorboard_write_freq == "batch":
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss,lambda=' +
                                  str(self.config.lambda_distillation),
                                  loss,
                                  step=step)
                tf.summary.scalar('loss_ce,lambda=' +
                                  str(self.config.lambda_distillation),
                                  output_loss,
                                  step=step)
                tf.summary.scalar('loss_distillation,lambda=' +
                                  str(self.config.lambda_distillation),
                                  distillation_loss,
                                  step=step)
                tf.summary.scalar('accuracy,lambda=' +
                                  str(self.config.lambda_distillation),
                                  self.acc_metric.result(),
                                  step=step)
            self.acc_metric.reset_states()
        elif self.config.tensorboard_write_freq == "epoch":
            self.loss_tracker.update_state(loss)
            self.loss_ce_tracker.update_state(output_loss)
            self.loss_distillation_tracker.update_state(distillation_loss)
        else:
            raise Exception("Invalid tensorboard_write_freq: %s!" %
                            self.config.tensorboard_write_freq)

    def test_step(self, test_x, test_y, test_m):
        """ Validating/Testing on one batch
            update losses & metrics
        """
        test_y_masked = tf.boolean_mask(test_y, test_m)
        if self.config.type_distillation == "feature":
            [pred_feature, pred_y] = self.new_model(test_x, training=False)
            pred_y_masked = tf.boolean_mask(pred_y, test_m)
            old_feature = self.old_encoder(test_x, training=False)
            distillation_loss = self.loss_mse(tf.stop_gradient(old_feature),
                                              pred_feature)
        elif self.config.type_distillation == "output":
            [_, pred_y] = self.new_model(test_x, training=False)
            pred_y_masked = tf.boolean_mask(pred_y, test_m)
            pseudo_y = self.old_model(test_x, training=False)
            pseudo_y = keras.backend.argmax(pseudo_y, axis=-1)
            pseudo_y_masked = tf.boolean_mask(pseudo_y, test_m)
            distillation_loss = self.loss_ce(tf.stop_gradient(pseudo_y_masked),
                                             pred_y_masked)
        output_loss = self.loss_ce(test_y_masked, pred_y_masked)
        loss = (
            1 - self.config.lambda_distillation
        ) * output_loss + self.config.lambda_distillation * distillation_loss
        pred_y = keras.backend.argmax(pred_y, axis=-1)
        pred_y_masked = tf.boolean_mask(pred_y, test_m)
        # Update val/test metrics
        self.loss_tracker.update_state(loss)
        self.loss_ce_tracker.update_state(output_loss)
        self.loss_distillation_tracker.update_state(distillation_loss)
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
            self.loss_distillation_tracker.reset_states()
            self.acc_metric.reset_states()

    def write_to_tensorboard(self, summary_writer, step):
        """
            write losses and metrics to tensorboard
        """
        with summary_writer.as_default():
            tf.summary.scalar('loss,lambda=' +
                              str(self.config.lambda_distillation),
                              self.loss_tracker.result(),
                              step=step)
            tf.summary.scalar('loss_ce,lambda=' +
                              str(self.config.lambda_distillation),
                              self.loss_ce_tracker.result(),
                              step=step)
            tf.summary.scalar('loss_distillation,lambda=' +
                              str(self.config.lambda_distillation),
                              self.loss_distillation_tracker.result(),
                              step=step)
            tf.summary.scalar('accuracy,lambda=' +
                              str(self.config.lambda_distillation),
                              self.acc_metric.result(),
                              step=step)
        self.loss_tracker.reset_states()
        self.loss_ce_tracker.reset_states()
        self.loss_distillation_tracker.reset_states()
        self.acc_metric.reset_states()


if __name__ == "__main__":
    experiment = Distillation()
    experiment.run()
