import segmentation_models as sm
import tensorflow as tf

from bfseg.utils.losses import ignorant_cross_entropy_loss, smooth_consistency_loss
from bfseg.utils.metrics import IgnorantAccuracyMetric

# import tf.keras.backend as K


def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0 / 10.0):
    # https://github.com/ialhashim/DenseDepth/blob/master/loss.py
    # Point-wise depth
    l_depth =  tf.keras.backend.mean( tf.keras.backend.abs(y_pred - y_true), axis=-1)

    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges =  tf.keras.backend.mean( tf.keras.backend.abs(dy_pred - dy_true) +  tf.keras.backend.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim =  tf.keras.backend.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta

    return (w1 * l_ssim) + (w2 *  tf.keras.backend.mean(l_edges)) + (w3 *  tf.keras.backend.mean(l_depth))

class MultiTaskModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(MultiTaskModel, self).__init__(*args, **kwargs)

        self.lossTracker = tf.keras.metrics.Mean(name="loss")
        self.semsegLossTracker = tf.keras.metrics.Mean(name="semsegLoss")
        self.depthLossTracker = tf.keras.metrics.Mean(name="depthLoss")
        self.consistencyLossTracker = tf.keras.metrics.Mean(name="consistencyLoss")
        #self.semsegAcc = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.semsegAcc = IgnorantAccuracyMetric()

        self.useIgnorantLosses = False


    def test_step(self, data):
        # only use semseg
        x, semseg_label = data

        if not tf.is_tensor(semseg_label):
            semseg_label = semseg_label['semseg']

        y_pred = self(x, training=False)  # Forward pass
        y_pred_depth = y_pred[0]
        y_pred_semseg = y_pred[1]
        y_pred_semseg_categorical = tf.expand_dims(tf.argmax(y_pred_semseg, axis=-1), axis=-1)

        if self.useIgnorantLosses:
            # Loss for SemSeg Error
            semsegLoss = ignorant_cross_entropy_loss(semseg_label, y_pred_semseg)
        else:
            # Loss for SemSeg Error
            semsegLoss = tf.keras.losses.sparse_categorical_crossentropy(semseg_label, y_pred_semseg)

        loss = semsegLoss


        # Compute our own metrics
        self.lossTracker.update_state(loss)
        self.semsegLossTracker.update_state(semsegLoss)
        self.semsegAcc(semseg_label, y_pred_semseg)

        return {"loss": self.lossTracker.result(), "semsegLoss": self.semsegLossTracker.result(),
                "accuracy": self.semsegAcc.result()}

    def train_step(self, data):
        x, y = data
        depth_label = y['depth']
        depth_label = depth_label
        semseg_label = y['semseg']

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            y_pred_depth = y_pred[0]
            y_pred_semseg = y_pred[1]
            y_pred_semseg_categorical = tf.expand_dims(tf.argmax(y_pred_semseg, axis = -1), axis = -1)
            # Loss for depth error, remove zero
            #y_pred_depth_ignorant = tf.math.multiply(y_pred_depth, tf.cast(tf.math.equal(depth_label, tf.constant(1000, dtype=tf.float32)), tf.float32))

            consistency_loss = smooth_consistency_loss(depth_label, y_pred_semseg_categorical, class_number=0) +  smooth_consistency_loss(depth_label, y_pred_semseg_categorical, class_number=1)

            y_pred_depth_ignorant = tf.where(tf.math.is_nan(depth_label), tf.zeros_like(depth_label), y_pred_depth)
            depth_label = tf.where(tf.math.is_nan(depth_label), tf.zeros_like(depth_label), depth_label)

            depthLoss = depth_loss_function(depth_label,y_pred_depth_ignorant)#tf.keras.losses.ber(depth_label, y_pred_depth_ignorant)

            if self.useIgnorantLosses:
                # Loss for SemSeg Error
                semsegLoss = ignorant_cross_entropy_loss(semseg_label, y_pred_semseg)
            else:
                # Loss for SemSeg Error
                semsegLoss = tf.keras.losses.sparse_categorical_crossentropy(semseg_label, y_pred_semseg)

            loss = depthLoss + semsegLoss + consistency_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.lossTracker.update_state(loss)
        self.semsegLossTracker.update_state(semsegLoss)
        self.depthLossTracker.update_state(depthLoss)
        self.consistencyLossTracker.update_state(consistency_loss)
        self.semsegAcc(semseg_label, y_pred_semseg)

        return {"loss": self.lossTracker.result(), "semsegLoss":self.semsegLossTracker.result(), "depthLoss": self.depthLossTracker.result(), "consistencyLoss": self.consistencyLossTracker.result(), "accuracy": self.semsegAcc.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.lossTracker, self.depthLossTracker,self.semsegLossTracker, self.consistencyLossTracker, self.semsegAcc]




def Deeplabv3(weights='pascal_voc', input_tensor=None, input_shape=(512, 512, 3), classes=21, backbone='xception',
              OS=8, alpha=1., activation=None):
    import bfseg.models.deeplabWithDepth as dlwd
    return dlwd.Deeplabv3(weights=weights, input_tensor=input_tensor, input_shape=input_shape, classes=classes,
                          backbone=backbone, OS=OS, alpha=alpha, activation=activation)


def PSPNet(backbone_name='vgg16',
           input_shape=(384, 384, 3),
           classes=21,
           activation='softmax',
           weights=None,
           encoder_weights='imagenet',
           encoder_freeze=False,
           downsample_factor=8,
           psp_conv_filters=512,
           psp_pooling_type='avg',
           psp_use_batchnorm=True,
           psp_dropout=None,
           **kwargs):
    model = sm.PSPNet(backbone_name=backbone_name, input_shape=input_shape, classes=classes, activation=activation,
                      weights=weights, encoder_weights=encoder_weights, encoder_freeze=encoder_freeze,
                      downsample_factor=downsample_factor, psp_conv_filters=psp_conv_filters,
                      psp_pooling_type=psp_pooling_type, psp_use_batchnorm=psp_use_batchnorm, psp_dropout=psp_dropout,
                      **kwargs)

    # Remove current decoder
    last_out = None
    for layer in model.layers:
        if (layer.name == "final_conv"):
            break
        last_out = layer.output

    depth = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer='glorot_uniform',
        name='final_conv_depth',
    )(last_out)

    semseg = tf.keras.layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer='glorot_uniform',
        name='final_conv_semseg',
    )(last_out)

    depth = tf.keras.layers.UpSampling2D(downsample_factor, name='depth',
                                         interpolation='bilinear')(depth)
    semseg = tf.keras.layers.UpSampling2D(downsample_factor, name='final_upsampling_semseg',
                                          interpolation='bilinear')(semseg)
    semseg = tf.keras.layers.Activation("softmax", name="semseg")(semseg)

    return MultiTaskModel(inputs=model.input, outputs=[depth, semseg])
