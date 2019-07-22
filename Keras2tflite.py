import os
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dropout, Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras import backend as K
from keras import optimizers
from tensorflow.contrib import lite
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.stats.mstats import gmean

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Manage memory 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

DATASET_PATH = "./Data/"
SAVE_PATH = "./Model/tflite/"
file_name = "CNN_keras_2.5s_Y50"

INPUT_SIGNAL_TYPES = [
    "body_wrist_acc_x",
    "body_wrist_acc_y",
    "body_wrist_acc_z",
    "body_wrist_rgacc_x",
    "body_wrist_rgacc_y",
    "body_wrist_rgacc_z",
    "body_wrist_gyro_x",
    "body_wrist_gyro_y",
    "body_wrist_gyro_z",
    "body_waist_acc_x",
    "body_waist_acc_y",
    "body_waist_acc_z",
    "body_waist_rgacc_x",
    "body_waist_rgacc_y",
    "body_waist_rgacc_z",
    "body_waist_gyro_x",
    "body_waist_gyro_y",
    "body_waist_gyro_z",
    "body_ankle_acc_x",
    "body_ankle_acc_y",
    "body_ankle_acc_z",
    "body_ankle_rgacc_x",
    "body_ankle_rgacc_y",
    "body_ankle_rgacc_z",
    "body_ankle_gyro_x",
    "body_ankle_gyro_y",
    "body_ankle_gyro_z"
]

LABELS = [
    "Sitting",
    "Standing", 
    "Walking", 
    "Going upstairs", 
    "Going downstairs", 
    "Drinking water",
    "Brushing teeth",
    "Cleaning", 
    "Jogging", 
    "Opening a door", 
    "Stretching", 
    "Lying down",
    "Walking while using a mobile phone"
] 

# Load "X" (the neural network's training and testing inputs)
def load_X(X_signals_paths):
    X_signals = []
    
    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()
        
    return np.transpose(np.array(X_signals), (1, 2, 0))

# Load "y" (the neural network's training and testing outputs)
def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]], 
        dtype=np.int32
    )
    file.close()
    
    # Substract 1 to each output class for friendly 0-based indexing 
    return y_ - 1

def one_hot(y_, n_classes=6):
    # Function to encode neural one-hot output labels from number indexes 
    # e.g.: 
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def create_model():
    model = Sequential()
    
    model.add(Convolution2D(
    batch_input_shape=(None, num_inputs, num_steps, 1),
    filters=10,
    kernel_size=[3, 8],
    strides=1,
    padding='valid',      
    ))

    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(
    pool_size=[1,6], 
    strides=[1,4], 
    padding='valid',    
    )) 

    model.add(Convolution2D(30, [3, 8], strides=1, padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D([1, 6], [1, 4], 'valid'))

    model.add(Flatten())
    
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

if __name__ == '__main__':
    X_train_signals_paths = [
        DATASET_PATH  + signal + ".txt" for signal in INPUT_SIGNAL_TYPES
    ]
    X_train = load_X(X_train_signals_paths)
    y_train_path = DATASET_PATH + "y.txt"
    y_train = load_y(y_train_path)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    epochs = 200
    num_classes = 13                 # Activity classes (13)
    batch_size = 300
    num_inputs = len(X_train[0][0])  # 27 input parameters per timestep, height
    num_steps = len(X_train[0])      # 313 timesteps per series, width

    X = (X_train.reshape(len(X_train), 1, num_steps, num_inputs)).astype(np.float32)
    X = X.transpose((0, 3, 2, 1))
    Y = one_hot(y_train, num_classes)
    X_test = (X_test.reshape(len(X_test), 1, num_steps, num_inputs)).astype(np.float32)
    X_test = X_test.transpose((0, 3, 2, 1))
    Y_test = one_hot(y_test, num_classes)

    keras_file = SAVE_PATH + file_name + ".h5"
    lite_file = SAVE_PATH + file_name + ".tflite"
    model = create_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    mc = ModelCheckpoint(keras_file, monitor='loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='loss', patience=20, verbose=1)
    model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), verbose=1, callbacks=[early_stopping, mc])

    model.save(keras_file)
    converter = lite.TFLiteConverter.from_keras_model_file(keras_file)
    tflite_model = converter.convert()
    open(lite_file, "wb").write(tflite_model)


    
    
    
    
    
