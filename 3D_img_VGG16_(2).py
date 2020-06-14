# -*- coding: cp936 -*-
# from __future__ import print_function
# import torch
from sched import scheduler

import sklearn as sk
import copy
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import shutil
import nibabel as nib
import threading
# import tensorlayer as tl
from matplotlib.pyplot import plot, title

import scipy
# from tensorlayer.prepro import *
import skimage.measure
import sklearn as sk
import sklearn.model_selection

from skimage.transform import resize
# from functools import reduce
# from utils import nice_print, mem_report, cpu_stats
# import copy
# import operator
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from eidetic_3d_lstm_net_8 import rnn
from VGG16 import VGG16
# from scipy import ndimage
from tensorflow.keras import models, optimizers, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Conv3D, MaxPooling3D, Activation
from tensorflow import keras
from keras import backend as K
import matplotlib.pyplot as plt

label_class = {'NC': 0, 'MCI': 1, 'AD': 2}


def switch(arg):
    switcher = {
        'AD': 0,  # stable NL
        'MCI': 1,  # stable MCI
        'CN': 2,  # stable AD
        'EMCI': 3,  # to MCI
        'SMCI': 4,  # to AD
        # def show_img(ori_img):
        'LMCI': 5,  # to AD
    }
    return switcher.get(arg, -1)


def read_data(path):
    image_data = nib.load(path).get_data()
    return image_data


def get_file_bypath(root_path):
    path_lists = os.listdir(root_path)
    for dir_file in path_lists:
        dir_file_path = os.path.join(root_path, dir_file)
        error_file_path = os.path.join(error_path, dir_file)

        if os.path.isdir(dir_file_path):
            get_file_bypath(dir_file_path)
        else:
            try:
                data = read_data(dir_file_path)
            except Exception as e:
                print(dir_file)
                shutil.move(dir_file_path, error_file_path)
            data = np.array(data)
            dataImg_list.append(data)
            data_ID.append(dir_file.split('_')[-1].split('.')[0])
            data_label.append(dir_file.split('_')[0])
            global data_list
            data_list = data_list.append({'ID': dir_file.split('_')[-1].split('.')[0],
                                          'ImagePath': dir_file_path,
                                          # 'Img':data,
                                          'label': dir_file.split('_')[0],
                                          'Sex': dir_file.split('_')[1],
                                          'Age': dir_file.split('_')[2],
                                          'Visit': dir_file.split('_')[3],
                                          'Subject': dir_file.split('_')[6] + "_" + dir_file.split('_')[7] + "_" +
                                                     dir_file.split('_')[8]}, ignore_index=True)
    # print(data_list.head())
    return data_list, data_ID, dataImg_list, data_label


def fetch_data(root_path):
    data_list, data_ID, dataImg_list, data_label = get_file_bypath(root_path)
    # data_list['label']= data_list['label'].apply(lambda x: switch(x)).values
    data_list['label'] = data_list['label'].apply(lambda x: switch(x)).values
    data_list = pd.DataFrame(data_list)
    data_list_filter = data_list[data_list['label'].isin([0, 1, 2])]
    # data_list_filter = data_list[data_list['label'].isin(['NC', 'MCI', 'AD'])]

    # data_list_filter[data_list['label']]= label_class.get(data_list_filter['label'])

    data_group_count = data_list_filter.groupby('Subject').count()

    Subject1 = data_group_count[data_group_count['Visit'] > 1].index
    print(Subject1)
    Subject2 = data_group_count[data_group_count['Visit'] == 1].index
    print(Subject2)

    global img_list1, label_list1, img_list2, label_list2
    # if  Subject1.ntnul :
    for i in Subject1:

        Subject = data_list_filter[data_list_filter['Subject'] == i].sort_values(by=['Visit'], ascending=True)
        img_img = []
        img_label = []
        for j in Subject['Visit']:
            Subject_img_path = Subject[Subject['Visit'] == j]
            Img_path = str(Subject_img_path['ImagePath'].values)
            data = read_data(Img_path[2:-2])
            data = resize(data, (256, 256, 256, 1))

            img_img.append(data)
            # img_img.append(data)
            img_label.append(Subject_img_path['label'].values)

        win = 2
        k = 0
        for k in range(len(img_img) - win + 1):
            # if k % win==0:
            input_img = tf.stack(img_img[k:k + win:], axis=0)
            input_label = tf.stack(img_label[k:k + win:], axis=0)
            img_list.append(input_img)
            label_list.append(input_label)

    img_list1 = tf.stack(img_list, axis=0)
    # img_list1=tf.squeeze(img_list1)
    label_list1 = tf.stack(label_list, axis=0)
    label_list1 = tf.squeeze(label_list1)

    label_list1 = K.one_hot(label_list1, label_type)
    label_list1 = K.eval(label_list1)

    for i in Subject2:
        Subject_img_path = data_list_filter[data_list_filter['Subject'] == i].sort_values(by=['Visit'], ascending=True)
        Img_path = str(Subject_img_path['ImagePath'].values)
        data = read_data(Img_path[2:-2])
        data = resize(data, (256, 256, 256, 1))
        img_list2.append(data)
        label_list2.append(Subject_img_path['label'].values)

    img_list2 = tf.stack(img_list2, axis=0)
    label_list2 = tf.squeeze(label_list2)
    label_list2 = K.one_hot(label_list2, label_type)
    label_list2 = K.eval(label_list2)

    return img_list1, label_list1, img_list2, label_list2


def scheduler(epoch):
    if epoch < epoch_num * 0.4:
        return learning_rate
    if epoch < epoch_num * 0.8:
        return learning_rate * 0.1
    return learning_rate * 0.01


dataPath_list = []
data_label = []
data_ID = []
dataImg_list = []
data_list = pd.DataFrame(columns=('ID', 'ImagePath', 'label', 'Sex', 'Age', 'Visit', 'Subject'))
# data_ID_Idx=[]
Subject_img_list = []
# root_path = './mapdcm2nii'
root_path = './train'
error_path = './error'

img_list = []
label_list = []
img_list1 = []
label_list1 = []
img_list2 = []
label_list2 = []
label_type = 3

img_list1, label_list1, img_list2, label_list2 = fetch_data(root_path)


# print(img_list1.shape)


class configs(object):
    def __init__(self, step_length, input_length):
        self.step_length = step_length
        self.input_length = input_length


step_length = img_list1.shape[1]
input_length = img_list1.shape[0]
configs1 = configs(step_length, input_length)
# flipped_images = tf.image.random_flip_left_right(tf.convert_to_tensor(img_list1))
# img_list1 = tf.convert_to_tensor(img_list1, dtype=tf.float32)
num_layers = 3
num_hidden = [256, 128, 64]
# loss = rnn(img_list1, label_list1, num_layers, num_hidden, configs1)
# print(' loss:' + loss)

weight_decay = 5e-4
batch_size = 2
learning_rate = 0.001
dropout_rate = 0.1
epoch_num = 20

data_size = img_list1.shape[0]
train_test_split = (int)(data_size * 0.2)
x1_train = img_list1[train_test_split:]
y1_train = label_list1[train_test_split:]
x1_test = img_list1[:train_test_split]
y1_test = label_list1[:train_test_split]

data_size = img_list2.shape[0]
train_test_split = (int)(data_size * 0.2)
x2_train = img_list2[train_test_split:]
y2_train = label_list2[train_test_split:]
x2_test = img_list2[:train_test_split]
y2_test = label_list2[:train_test_split]

model = VGG16()
model.summary()
sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
change_lr = LearningRateScheduler(scheduler(10))

data_augmentation = False

if not data_augmentation:
    print('Not using data augmentation')
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # checkpoint_filepath = 'models/model-ep{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5'
    # checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

    model.summary()

    history = model.fit(x2_train, y2_train, batch_size=batch_size, steps_per_epoch=2, epochs=epoch_num,
                        callbacks=[change_lr])

# model.fit(x_train, y_train, batch_size=batch_size, steps_per_epoch=2, epochs=200,validation_data=(x_test, y_test))
else:
    print('Using real-time data augmentation')
    #  train_datagen= keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,rotation_range=10,width_shift_range=0.1,height_shift_range=0.1,
    # train_datagen = keras.preprocessing.image.ImageDataGenerator()                                                          # shear_range=0.1,zoom_range=0.1,horizontal_flip=False,fill_mode='neares

    # test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    # test_datagen = keras.preprocessing.image.ImageDataGenerator()

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # model.fit(train_datagen.flow(x_train,y_train, batch_size=32),steps_per_epoch=x_train.shape[0],epochs=200, callbacks=[change_lr],validation_data=test_datagen.flow(x_test,y_test, batch_size=32),validation_steps=x_test.shape[0])
    model.fit(x2_train, y2_train, batch_size=32, validation_split=0.2, steps_per_epoch=x2_train.shape[0], epochs=200)
    print("hello")
validation_steps = 2
loss0, accuracy0 = model.evaluate(x2_test, y2_test, steps=validation_steps)
print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))
plot(accuracy0)
plot(loss0)
plot(history.history['loss'])

title('model accuracy')
