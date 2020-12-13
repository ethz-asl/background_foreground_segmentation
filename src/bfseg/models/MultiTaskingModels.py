import segmentation_models as sm
import tensorflow as tf

from bfseg.utils.losses import ignorant_cross_entropy_loss


class MultiTaskModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(MultiTaskModel, self).__init__(*args, **kwargs)

        self.semsegLossTracker = tf.keras.metrics.Mean(name="semsegLoss")
        self.depthLossTracker = tf.keras.metrics.Mean(name="depthLoss")
        self.semsegAcc = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

        self.useIgnorantLosses = False

    def train_step(self, data):
        x, y = data
        depth_label = y['depth']
        depth_label = depth_label
        semseg_label = y['semseg']

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            y_pred_depth = y_pred[0]
            y_pred_semseg = y_pred[1]

            # Loss for depth error, remove zero
            y_pred_depth_ignorant = tf.math.multiply(y_pred_depth, tf.cast(tf.math.equal(depth_label, tf.constant(1000, dtype=tf.float32)), tf.float32))

            depthLoss = tf.keras.losses.mean_squared_error(depth_label, y_pred_depth_ignorant)

            if self.useIgnorantLosses:
                # Loss for SemSeg Error
                semsegLoss = ignorant_cross_entropy_loss(semseg_label, y_pred_semseg)
            else:
                # Loss for SemSeg Error
                semsegLoss = tf.keras.losses.sparse_categorical_crossentropy(semseg_label, y_pred_semseg)

            loss = depthLoss + semsegLoss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.semsegLossTracker.update_state(semsegLoss)
        self.depthLossTracker.update_state(depthLoss)
        self.semsegAcc(semseg_label, y_pred_semseg)

        return {"semsegLoss":self.semsegLossTracker.result(), "depthLoss": self.depthLossTracker.result(), "accuracy": self.semsegAcc.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.depthLossTracker,self.semsegLossTracker, self.semsegAcc]




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
