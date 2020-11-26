###########################################################################################
#    Contains all tensorboard functionality
###########################################################################################
import os
from tensorboard import program

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from datetime import datetime


def plotImages(epoch, logs, model, validation_ds, file_writer):
    # Use the model to predict the values from the validation dataset.
    for img, label in validation_ds.take(1):
        test_pred_raw = model.predict(img)
        pred_mask = tf.argmax(test_pred_raw, axis=3)
        pred_mask = pred_mask[..., tf.newaxis]

        with file_writer.as_default():
            tf.summary.image("Valid_img", img, step=0)

        with file_writer.as_default():
            tf.summary.image("Valid_gt", label / 2, step=0)

        with file_writer.as_default():
            tf.summary.image("Valid_predict", pred_mask / 2, step=0)


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


class TensorBoardImage(tf.keras.callbacks.Callback):
    def __init__(self, tag, training_ds, validation_ds, model):
        super().__init__()
        self.tag = tag
        self.validation_ds = validation_ds
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        for img, label in self.validation_ds.take(1):
            test_pred_raw = self.model.predict(img)
            pred_mask = tf.argmax(test_pred_raw, axis=-1)
            pred_mask = pred_mask[..., tf.newaxis]

            file_writer = tf.summary.create_file_writer("./logs")

            with file_writer.as_default():
                tf.summary.image("Valid_img", img, step=epoch)

            with file_writer.as_default():
                tf.summary.image("Valid_predict", tf.cast(pred_mask, tf.float32), step=epoch)

            with file_writer.as_default():
                tf.summary.image("Valid_gt", label / 2, step=epoch)



class TensorBoardMonitor:
    def __init__(self, training_ds, validation_ds, model, log_dir="./logs"):
        self.callbacks = []
        self.callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

        logdir = log_dir + "/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # Creates a file writer for the log directory.
        file_writer = tf.summary.create_file_writer(logdir)

        def callback_fn(epoch, logs):
            plotImages(epoch, logs, model=model, validation_ds=validation_ds, file_writer=file_writer)

        # Define the per-epoch callback.
        self.callbacks.append(TensorBoardImage("Test Images", training_ds, validation_ds, model))

    def startTensorboard(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', "logs"])
        url = tb.launch()
        print("tensorboard running on ", url)

    def getCallbacks(self):
        return self.callbacks
