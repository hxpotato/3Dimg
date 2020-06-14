import pandas as pd
import numpy as np
import tensorflow as tf
# import tensorflow.keras as keras
from tensorflow.keras import models, optimizers, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Conv3D, MaxPooling3D, Activation
from tensorflow import keras
import matplotlib.pyplot as plt

weight_decay = 5e-4
batch_size = 16
learning_rate = 0.001
dropout_rate = 0.1
epoch_num = 150


def VGG16():
    model = models.Sequential()
    model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=(256, 256, 256, 1),
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1)))

    model.add(
        Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None))

    model.add(
        Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(
        Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None))

    model.add(
        Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(
        Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None))

    model.add(
        Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(
        Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None))
    #
    model.add(Flatten())  # 2*2*512
    model.add(Dense(4068, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    return model
