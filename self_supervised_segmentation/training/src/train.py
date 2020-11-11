###########################################################################################
#    Main entry point to train model
###########################################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import modelBuilder 
from dataLoader import DataLoader
from tbMonitor import TensorBoardMonitor

def main():    
    workingdir = "/home/rene/cla_dataset/cam0_preprocessed/"
    dataLoader = DataLoader(workingdir, [240, 480])
    ds_train, ds_valid = dataLoader.getDataset()

    mdl = modelBuilder.getModel()
    mdl.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    mdl.summary()

    monitor = TensorBoardMonitor(ds_train, ds_valid, mdl)
    monitor.startTensorboard()

    mdl.fit(ds_train, epochs=2, steps_per_epoch=1, callbacks = monitor.getCallbacks())




if __name__ == "__main__":
    main()
    input("Press enter to stop")

    