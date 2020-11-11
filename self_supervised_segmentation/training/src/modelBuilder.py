###########################################################################################
#    Contains all code that is used to generate a model
###########################################################################################
import tensorflow as tf

def getModel(input_shape = (240,480,3)):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(3, 3, padding="same", activation='relu', input_shape= input_shape))
    return model