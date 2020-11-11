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
            tf.summary.image("Valid_img", img, step = 0)

        with file_writer.as_default():
            tf.summary.image("Valid_gt", label/2, step = 0)
            
        with file_writer.as_default():
            tf.summary.image("Valid_predict", pred_mask/2, step = 0)



class TensorBoardMonitor:
    def __init__(self, training_ds, validation_ds, model, log_dir = "./logs"): 
        self.callbacks = []
        self.callbacks.append(tf.keras.callbacks.TensorBoard(log_dir= log_dir, histogram_freq=1))


        logdir = log_dir + "/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # Creates a file writer for the log directory.
        file_writer = tf.summary.create_file_writer(logdir)
        
        def callback_fn(epoch, logs):
            plotImages(epoch, logs, model = model, validation_ds = validation_ds, file_writer = file_writer)
        
        # Define the per-epoch callback.
        self.callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=callback_fn))


    def startTensorboard(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', "logs"])
        url = tb.launch()
        print("tensorboard running on ", url)

    def getCallbacks(self):
        return self.callbacks
