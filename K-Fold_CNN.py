import os
import time 
import numpy as np
import tensorflow as tf
from tensorflow.contrib import lite
from keras.models import Sequential, load_model, Model
from keras.layers import Dropout, Dense, Activation, Convolution2D, MaxPooling2D, Flatten, concatenate, Input
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras import optimizers
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import gmean

# Manage memory 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

DATASET_PATH = "./Data/"
SAVE_PATH = "./Model/10fold/"

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
    "Going_upstairs", 
    "Going_downstairs", 
    "Drinking_water",
    "Brushing_teeth",
    "Cleaning", 
    "Jogging", 
    "Opening_a_door", 
    "Stretching", 
    "Lying_down",
    "Walking_while_using_a_mobile_phone"
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

# Function to encode neural one-hot output labels from number indexes 
def one_hot(y_, n_classes=6):
    # e.g.: 
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def mcor(y_true, y_pred):
    #matthews_correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
 
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
 
    return numerator / (denominator + K.epsilon())

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

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

    # load sample data
    X_train_signals_paths = [
        DATASET_PATH  + signal + ".txt" for signal in INPUT_SIGNAL_TYPES
    ]
    X_train = load_X(X_train_signals_paths)

    # load label data
    y_train_path = DATASET_PATH + "y.txt"
    y_train = load_y(y_train_path)

    # parameters
    epochs = 200
    num_classes = 13                 # Activity classes (13)       
    batch_size = 300
    num_inputs = len(X_train[0][0])  # 27 input parameters per timestep, height
    num_steps = len(X_train[0])      # 313 timesteps per series(2.5s), width

    # reshape 
    X = (X_train.reshape(len(X_train), 1, num_steps, num_inputs)).astype(np.float32)
    X = X.transpose((0, 3, 2, 1))
    Y = one_hot(y_train, num_classes)

    # divided data to 10 partitions 
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # for statistic
    LossList = []
    AccList = []
    McorList = []
    PrecList = []
    RecallList = []
    F1List = []

    i = 0
    # The for-loop will do 10 times because of line 264
    for train_index, test_index in kfold.split(X, y_train):
        keras_file = SAVE_PATH + "model_i" + str(i) + "_best.h5"
        # --------- shuffle data -----------
        train_X, valid_X, train_y, valid_y = train_test_split(X[train_index], Y[train_index], test_size = 0.2, random_state = 42)
        test_X, test_y = shuffle(X[test_index], Y[test_index], random_state = 42)

        # --------- craete CNN model ---------
        model = create_model()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', mcor, precision, recall, f1])
    
        # --------- train ---------
        tStart = time.time()
        # --------- training and validation ---------
        early_stopping = EarlyStopping(monitor='loss', patience=20, verbose=0)
        mc = ModelCheckpoint(keras_file, monitor='loss', verbose=0, save_best_only=True)
        trainHistory = model.fit(train_X, train_y, epochs=epochs, batch_size = batch_size, verbose = 0, validation_data=(valid_X, valid_y), callbacks=[early_stopping, mc])
        train_loss = trainHistory.history['loss']
        val_loss   = trainHistory.history['val_loss']
        train_acc  = trainHistory.history['acc']
        val_acc    = trainHistory.history['val_acc']
        tEnd = time.time()
        print("Train time: %f sec" % (tEnd-tStart))
        # --------- end training ---------
    
        # --------- evaluate accuracy from the best epoch ---------
        model = load_model(SAVE_PATH + "model_i" +str(i) + "_best.h5", custom_objects={'mcor':mcor, 'precision':precision, 'recall':recall, 'f1':f1})
        loss_, acc_, mcor_, precision_, recall_, f1_ = model.evaluate(test_X, test_y, batch_size=batch_size, verbose=False)
        predictions = model.predict_classes(test_X)
        cm = metrics.confusion_matrix(np.argmax(test_y, 1), predictions,)
        LossList.append(loss_)
        AccList.append(acc_)
        McorList.append(mcor_)
        PrecList.append(precision_)
        RecallList.append(recall_)
        F1List.append(f1_)
        i += 1
    print("meanLoss:%lf, meanAcc:%lf, meanMcor:%lf, meanPrecision:%lf, meanRecall:%lf, meanF1:%lf" %(gmean(LossList), gmean(AccList), gmean(McorList), gmean(PrecList), gmean(RecallList), gmean(F1List)))

