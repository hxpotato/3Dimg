# -*- coding: cp936 -*-
# from __future__ import print_function
# import torch
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
import matplotlib.pyplot as plt

import scipy
# from tensorlayer.prepro import *
import skimage.measure
import sklearn as sk
import sklearn.model_selection

from skimage.transform import resize
# from functools import reduce
# from utils import nice_print, mem_report, cpu_stats
# import copy
# import operator                      c
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from eidetic_3d_lstm_net_8 import rnn
from scipy import ndimage
def show_img(ori_img):
    plt.imshow(ori_img[:, :, 25])
    plt.show()


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


# def read_img_list(Subject,data_list_filter):
# for i in Subject:
# Subject= data_list_filter[data_list_filter['Subject']==i].sort_values(by=['Visit'], ascending=True)
# for j in Subject['Visit']:
# Subject_img_path= Subject[Subject['Visit']==j]
# Img_path =str(Subject_img_path['ImagePath'].values)
# data=read_data(Img_path[2:-2])
# data=resize(data,(256,256,256))
# img_list.append(data)
# return img_list

def fetch_data(root_path):
    data_list, data_ID, dataImg_list, data_label = get_file_bypath(root_path)
    data_list_filter = data_list[data_list['label'].isin(['AD', 'MCI', 'CN'])]
    data_group_count = data_list_filter.groupby('Subject').count()

    Subject1 = data_group_count[data_group_count['Visit'] > 1].index
    print(Subject1)
    Subject2 = data_group_count[data_group_count['Visit'] == 1].index
    print(Subject2)

    # img_list1=read_img_list(Subject1,data_list_filter)
    # img_list2=read_img_list(Subject2,data_list_filter)

    global img_list1, label_list1, img_list2, label_list2
    for i in Subject1:

        Subject = data_list_filter[data_list_filter['Subject'] == i].sort_values(by=['Visit'], ascending=True)
        #  Subject= Subject.sort_values(by=['Visit'],axis=0,ascending=True)
        # k=0
        img_img = []
        img_label = []
        for j in Subject['Visit']:
            Subject_img_path = Subject[Subject['Visit'] == j]
            Img_path = str(Subject_img_path['ImagePath'].values)
            data = read_data(Img_path[2:-2])
            data = resize(data, (256, 256, 256,1))
            #data=tf.shape(data,[256, 256, 256])
            #data=ndimage.zoom(data,(256,256,256),order=1)
            img_img.append(data)
            #img_img.append(data)
            img_label.append(Subject_img_path['label'].values)

        #img_list = np.array(img_img)
        #img_list1.append(img_list)
        #img_img=np.array(img_img)
        win=2
        k=0
        for k in range(len(img_img)-win+1):
             #if k % win==0:
                input_img = tf.stack(img_img[k:k+win:],axis=0)
                input_label= tf.stack(img_label[k:k+win:],axis=0)


                img_list.append(input_img)
                label_list.append(input_label)

       # img_label=np.array(img_label)
       # img_label1.append(img_label)

    img_list1=tf.stack(img_list,axis=0)
    img_list1=tf.squeeze(img_list1)
    label_list1=tf.stack(label_list,axis=0)
    label_list1=tf.squeeze(label_list1)
    #img_list1=np.array(img_list)
    #img_label1 = np.array(img_label)
        # img_list=np.array(img_list)
        # k=k+1
    #img_list1.append(img_list)
    #img_list1=np.array(img_list1)
    # img_list1=img_list1.reshape((256,256,256,2))
    # return Subject_img_list1
    for i in Subject2:
        Subject = data_list_filter[data_list_filter['Subject'] == i].sort_values(by=['Visit'], ascending=True)
        Img_path = str(Subject_img_path['ImagePath'].values)
        data = read_data(Img_path[2:-2])
        data = resize(data, (256, 256, 256,1))
        data=tf.squeeze (data)
        img_list2.append(data)
        label_list2.append(Subject_img_path['label'].values)
    # return Subject_img_list1
    # img_list1 =resize(img_list1,(2,1,256,256,256))
   # img_list = np.array(img_list1)
    #img_list1 = resize(img_list1, (2, 1, 256, 256, 256))
    #img_label1 = np.array(img_label1)

    #img_list2 = np.array(img_list2)
    #label_list2 = np.array(label_list2)

    img_list2 = tf.stack(img_list2,axis=0)
    label_list2 = tf.stack(label_list2,axis=0)

    return img_list1, label_list1, img_list2, label_list2


dataPath_list = []
data_label = []
data_ID = []
dataImg_list = []
data_list = pd.DataFrame(columns=('ID', 'ImagePath', 'label', 'Sex', 'Age', 'Visit', 'Subject'))
# data_ID_Idx=[]
Subject_img_list = []
# root_path = './mapdcm2nii'
root_path = './test'
error_path = './error'

img_list=[]
label_list=[]
img_list1 = []
label_list1=[]
img_list2 = []
label_list2=[]

img_list1, label_list1, img_list2, label_list2= fetch_data(root_path)
# img_list1=np.array(img_list1)
# img_list_S=img_list_S.reshape((256,256,256,1,4))

print(img_list1.shape)


# print(img_list1)
# print(img_list2)

# rnn(images, real_input_flag, num_layers, num_hidden, configs):

# print(img_list2)
class configs(object):
    def __init__(self, total_length, input_length):
        self.total_length = total_length
        self.input_length = input_length


total_length = tf.size(img_list1)
input_length = tf.shape(img_list1)[0]
configs1 = configs(total_length, input_length)
# flipped_images = tf.image.random_flip_left_right(tf.convert_to_tensor(img_list1))
#img_list1 = tf.convert_to_tensor(img_list1, dtype=tf.float32)
num_layers = 3
num_hidden = [256, 128, 64]
out_ims, loss = rnn(img_list1, label_list1, num_layers, num_hidden, configs1)
print('out_ims：' + out_ims + ' loss:' + loss)
