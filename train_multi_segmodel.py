import tensorflow as tf
import tensorflow_datasets as tfds
from Nyu_depth_v2_labeled.Nyu_depth_v2_labeled import NyuDepthV2Labeled 
import segmentation_models as sm

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

def main():
	train_ds, train_info = tfds.load('NyuDepthV2Labeled', split='train[:80%]',shuffle_files=True, as_supervised=True, with_info=True,)
	test_ds, test_info = tfds.load('NyuDepthV2Labeled', split='train[80%:]',shuffle_files=False, as_supervised=True, with_info=True,)
	train_ds = train_ds.map(
	    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	train_ds = train_ds.cache()
	train_ds = train_ds.shuffle(train_info.splits['train[:80%'].num_examples)
	train_ds = train_ds.batch(128)
	train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

	test_ds = test_ds.map(
	    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	test_ds = test_ds.batch(128)
	test_ds = test_ds.cache()
	test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

	BACKBONE = "vgg16"
	# preprocess_input = sm.get_preprocessing(BACKBONE)

	model = sm.PSPNet(BACKBONE,input_shape=(480, 640, 3),classes=894)
	
	model.compile(
	    loss='sparse_categorical_crossentropy',
	    optimizer=tf.keras.optimizers.Adam(0.001),
	    metrics=['accuracy'],
	)

	model.fit_generator(
	    train_ds,
	    epochs=6,
	    validation_data=test_ds,
	)
	model.load_weights('best_model.h5') 
	
if __name__ == "__main__": 
	main()