import tensorflow as tf
import tensorflow_datasets as tfds
from Nyu_depth_v2_labeled.Nyu_depth_v2_labeled import NyuDepthV2Labeled
import segmentation_models as sm
from tensorflow import keras
import tensorflow.keras.preprocessing.image as Image
import os
import numpy as np


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


def main():
  batch_size = 8
  step = "step2"
  if step == "step1":
    train_ds, train_info = tfds.load(
        'NyuDepthV2Labeled',
        split='train[:80%]',
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    val_ds, val_info = tfds.load(
        'NyuDepthV2Labeled',
        split='train[80%:]',
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    test_ds, test_info = tfds.load(
        'NyuDepthV2Labeled',
        split='test',
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )
    lr = 0.0001
    encoder_freezed = False
  else:
    train_ds, train_info = tfds.load(
        'NyuDepthV2Labeled',
        split='test[:80%]',
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    val_ds, val_info = tfds.load(
        'NyuDepthV2Labeled',
        split='test[80%:]',
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    test_ds, test_info = tfds.load(
        'NyuDepthV2Labeled',
        split='train[80%:]',
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )
    lr = 0.00001
    encoder_freezed = True

  train_ds = train_ds.map(normalize_img,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_ds = train_ds.cache()
  # train_ds = train_ds.shuffle(int(train_info.splits['train'].num_examples *
  # 0.8))
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

  feature_save_dir = "./bfseg/data/old_features"
  model_save_dir = os.path.join("./saved_model", "step2_new")
  try:
    os.makedirs(feature_save_dir)
    os.makedirs(model_save_dir)
  except os.error:
    pass

  BACKBONE = "vgg16"
  encoder, model = sm.Unet(
      BACKBONE,
      input_shape=(480, 640, 3),
      classes=2,
      activation='sigmoid',
      weights="./saved_model/step1/model.17-0.33.h5",
      # encoder_weights=None,
      encoder_freeze=False)
  New_model = keras.Model(inputs=model.input,
                          outputs=[encoder.output, model.output])
  # old_encoder, old_model = sm.Unet(BACKBONE,
  #                 input_shape=(480, 640, 3),
  #                 classes=2,
  #                 activation='sigmoid',
  #                 weights="./saved_model/step1/model.17-0.33.h5",
  #                 # encoder_weights=None,
  #                 encoder_freeze=True
  #                 )
  # old_model.trainable=False

  # if step == "step2":
  #   model.load_weights("./saved_model/step1/model.17-0.33.h5")
  # model.summary()
  # old_feature = old_model.get_layer("block5_conv3").output
  # new_feature = model.get_layer("block5_conv3").output
  # model.add_loss(tf.keras.losses.MSE(old_encoder(model.input),encoder(model.input)))
  # model.compile(
  #     loss='sparse_categorical_crossentropy',
  #     # loss=sm.losses.CategoricalFocalLoss(),
  #     optimizer=tf.keras.optimizers.Adam(lr),
  #     # metrics=metrics,
  #     metrics='accuracy',
  # )
  optimizer = keras.optimizers.Adam(lr)
  loss_ce = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
  loss_mse = keras.losses.MeanSquaredError()
  train_acc_metric = keras.metrics.Accuracy()
  val_acc_metric = keras.metrics.Accuracy()
  test_acc_metric = keras.metrics.Accuracy()
  epochs = 20
  for epoch in range(epochs):
    i = 0
    print("\nStart of epoch %d" % (epoch,))
    for image, label in train_ds:
      with tf.GradientTape() as tape:
        [pred_feature, pred_label] = New_model(image, training=True)
        output_loss = loss_ce(label, pred_label)
        # pred_feature = encoder(image, training=True)
        # old_feature = old_encoder(image, training=False)
        old_feature = tf.convert_to_tensor(
            np.load(feature_save_dir+"/batch_" + str('{:03d}'.format(i) + ".npy")),
            dtype=tf.float32)
        feature_loss = loss_mse(old_feature, pred_feature)
        loss = output_loss + feature_loss
      grads = tape.gradient(loss, New_model.trainable_weights)
      optimizer.apply_gradients(zip(grads, New_model.trainable_weights))
      # Update training metric.
      pred_label = keras.backend.argmax(pred_label, axis=-1)
      train_acc_metric.update_state(label, pred_label)
      # Log every 5 batches.
      if i % 5 == 0:
        print(
            "Total training/ output / feature loss at step %d: %.4f,%.4f,%.4f" %
            (i, float(loss), float(output_loss), float(feature_loss)))
      i = i + 1
      if i > 37:
        break
    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()
    # Run a validation loop at the end of each epoch.
    for image, label in val_ds:
      [pred_feature, pred_label] = New_model(image, training=False)
      # Update val metrics
      pred_label = keras.backend.argmax(pred_label, axis=-1)
      val_acc_metric.update_state(label, pred_label)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    for image, label in test_ds:
      [pred_feature, pred_label] = New_model(image, training=False)
      # Update val metrics
      pred_label = keras.backend.argmax(pred_label, axis=-1)
      test_acc_metric.update_state(label, pred_label)
    test_acc = test_acc_metric.result()
    test_acc_metric.reset_states()
    print("Testing acc: %.4f" % (float(test_acc),))
  # callbacks = [
  #   tf.keras.callbacks.ModelCheckpoint(os.path.join(model_save_dir,'model.{epoch:02d}-{val_loss:.2f}.h5'),
  #                                   save_weights_only=True,
  #                                   save_best_only=False,
  #                                   mode='min'),
  #   tf.keras.callbacks.ReduceLROnPlateau(),
  #   tf.keras.callbacks.TensorBoard(log_dir=os.path.join('./logs',"step2_new"),update_freq=5)
  # ]

  # history = model.fit(
  #     train_ds,
  #     steps_per_epoch=train_info.splits['train'].num_examples*0.8 // batch_size,
  #     epochs=50,
  #     validation_data=test_ds,
  #     callbacks=callbacks,
  # )

  # print("Evaluate on val data")
  # val_results = model.evaluate(val_ds)
  # print("val loss, val acc:", val_results)
  # print("Evaluate on test data")
  # test_results = model.evaluate(test_ds)
  # print("test loss, test acc:", test_results)

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
