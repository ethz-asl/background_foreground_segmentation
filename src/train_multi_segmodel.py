import tensorflow as tf
import tensorflow_datasets as tfds
from Nyu_depth_v2_labeled.Nyu_depth_v2_labeled import NyuDepthV2Labeled
import segmentation_models as sm
from tensorflow import keras
import tensorflow.keras.preprocessing.image as Image
import os


@tf.function
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  # image = tf.image.resize(image,(32,64))
  # label = tf.cast(label, tf.uint8)
  # combine_label = tf.cast(tf.math.logical_or(tf.math.equal(label,4),(tf.logical_or(tf.math.equal(label,11),tf.math.equal(label,21)))),tf.uint8)
  # combine_label= tf.expand_dims(combine_label,axis=2)
  label = tf.expand_dims(label, axis=2)
  # print(label.shape)
  image = tf.cast(image, tf.float32) / 255.
  # return image, combine_label
  return image, label


def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask


def main():
  batch_size = 8
  # train_ds, train_info = tfds.load('NyuDepthV2Labeled', split='train[:80%]',shuffle_files=True, as_supervised=True, with_info=True,)
  # test_ds, test_info = tfds.load('NyuDepthV2Labeled', split='train[80%:]',shuffle_files=False, as_supervised=True, with_info=True,)
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
  train_ds = train_ds.map(normalize_img,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_ds = train_ds.cache()
  train_ds = train_ds.shuffle(int(train_info.splits['train'].num_examples *
                                  0.8))
  # train_ds = train_ds.shuffle(int(train_info.splits['train'].num_examples*0.8))
  train_ds = train_ds.batch(batch_size).repeat()
  train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

  val_ds = val_ds.map(normalize_img,
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  val_ds = val_ds.batch(batch_size)
  val_ds = val_ds.cache()

  test_ds = test_ds.map(normalize_img,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  test_ds = test_ds.batch(batch_size)
  test_ds = test_ds.cache()
  # test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
  callbacks = [
      keras.callbacks.ModelCheckpoint('./best_model.h5',
                                      save_weights_only=True,
                                      save_best_only=True,
                                      mode='min'),
      keras.callbacks.ReduceLROnPlateau(),
  ]
  # for batch in train_ds:
  # 	print(batch[1])
  BACKBONE = "vgg16"
  # preprocess_input = sm.get_preprocessing(BACKBONE)

  model = sm.Unet(BACKBONE,
                  input_shape=(480, 640, 3),
                  classes=2,
                  activation='sigmoid')
  # model = tf.keras.models.Sequential([
  # 		tf.keras.layers.Flatten(input_shape=(48, 64, 3)),
  # 		tf.keras.layers.Dense(1024,activation='relu'),
  # 		tf.keras.layers.Dense(894, activation='softmax')
  # 		])
  model.summary()
  # metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
  model.compile(
      loss='sparse_categorical_crossentropy',
      # loss=sm.losses.CategoricalFocalLoss(),
      optimizer=tf.keras.optimizers.Adam(0.0001),
      # metrics=metrics,
      metrics='accuracy',
  )

  model.fit(
      train_ds,
      # steps_per_epoch=int(train_info.splits['train'].num_examples*0.8)//batch_size,
      steps_per_epoch=train_info.splits['train'].num_examples // batch_size,
      epochs=10,
      validation_data=test_ds,
      callbacks=callbacks,
  )
  model_save_dir = './checkpoints'
  if os.path.exists(model_save_dir) == False:
    os.mkdir(model_save_dir)
  model.save_weights(
      os.path.join(model_save_dir, 'model_epoch10_diffscene.ckpt'))
  print("Evaluate on val data")
  val_results = model.evaluate(val_ds)
  print("val loss, val acc:", val_results)
  print("Evaluate on test data")
  test_results = model.evaluate(test_ds)
  print("test loss, test acc:", test_results)

  testing_save_path = './test_result_epoch10_diffscene/'
  validating_save_path = './val_result_epoch10_diffscene/'
  if os.path.exists(testing_save_path) == False:
    os.mkdir(testing_save_path)
  if os.path.exists(validating_save_path) == False:
    os.mkdir(validating_save_path)
  i = 0
  for image, label in test_ds:
    for j in range(image.shape[0]):
      pred_label = model.predict(image)
      Image.save_img(
          os.path.join(testing_save_path,
                       str(i * batch_size + j).zfill(4) + '_image.png'),
          image[j])
      Image.save_img(
          os.path.join(testing_save_path,
                       str(i * batch_size + j).zfill(4) + '_trueseg.png'),
          label[j])
      # print(label[j])
      Image.save_img(
          os.path.join(testing_save_path,
                       str(i * batch_size + j).zfill(4) + '_preddeg.png'),
          create_mask(pred_label[j]))
      # print(create_mask(pred_label[j]))
    print("Predicting test batch %d" % i)
    i = i + 1
  i = 0
  for image, label in val_ds:
    for j in range(image.shape[0]):
      pred_label = model.predict(image)
      Image.save_img(
          os.path.join(validating_save_path,
                       str(i * batch_size + j).zfill(4) + '_image.png'),
          image[j])
      Image.save_img(
          os.path.join(validating_save_path,
                       str(i * batch_size + j).zfill(4) + '_trueseg.png'),
          label[j])
      Image.save_img(
          os.path.join(validating_save_path,
                       str(i * batch_size + j).zfill(4) + '_preddeg.png'),
          create_mask(pred_label[j]))
    print("Predicting train batch %d" % i)
    i = i + 1


if __name__ == "__main__":
  sm.set_framework('tf.keras')
  main()
