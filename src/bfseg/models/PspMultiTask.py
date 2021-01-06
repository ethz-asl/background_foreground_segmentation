import segmentation_models as sm
import tensorflow as tf


def PSPNetMultiTask(backbone_name='vgg16',
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

  pspModel = sm.PSPNet(backbone_name=backbone_name,
                       input_shape=input_shape,
                       classes=classes,
                       activation=activation,
                       weights=weights,
                       encoder_weights=encoder_weights,
                       encoder_freeze=encoder_freeze,
                       downsample_factor=downsample_factor,
                       psp_conv_filters=psp_conv_filters,
                       psp_use_batchnorm=psp_use_batchnorm,
                       psp_dropout=psp_dropout,
                       **kwargs)

  if psp_dropout is None:
    before_decoder = pspModel.get_layer("aggregation_relu").output
  else:
    before_decoder = pspModel.get_layer("spatial_dropout").output

  depth = simpleDecoder(before_decoder, 1, activation, downsample_factor,
                        "depth")
  semseg = simpleDecoder(before_decoder, classes, activation, downsample_factor,
                         "semseg")

  # Pseudo output only used for smooth consistency loss
  combined = tf.keras.layers.concatenate([depth, semseg],
                                         axis=-1,
                                         name="combined")

  return tf.keras.models.Model(inputs=pspModel.input,
                               outputs=[depth, semseg, combined],
                               name="pspNetMultiTask")


def simpleDecoder(input, classes, activation, final_upsampling_factor, name):
  # model head
  x = tf.keras.layers.Conv2D(
      filters=classes,
      kernel_size=(3, 3),
      padding='same',
      kernel_initializer='glorot_uniform',
      name='final_conv' + name,
  )(input)

  x = tf.keras.layers.UpSampling2D(final_upsampling_factor,
                                   name='final_upsampling' + name,
                                   interpolation='bilinear')(x)
  x = tf.keras.layers.Activation(activation, name=name)(x)
  return x


if __name__ == '__main__':
  PSPNetMultiTask()
