from __future__ import print_function
import torch
import sklearn as sk
import numpy as np
import os
import shutil
import nibabel as nib
import threading
import tensorlayer as tl
import matplotlib.pyplot as plt
import scipy
from tensorlayer.prepro import *
import skimage.measure
import sklearn as sk
import sklearn.model_selection



def show_img(ori_img):
    plt.imshow(ori_img[:,:,25])
    plt.show()

def read_data(path):
    image_data = nib.load(path).get_data()
    return image_data

def get_file_bypath(root_path):
    path_lists = os.listdir(root_path)
    for dir_file in path_lists:
        dir_file_path = os.path.join(root_path,dir_file)
        error_file_path= os.path.join(error_path,dir_file)

        if os.path.isdir(dir_file_path):
           get_file_bypath(dir_file_path)
        else:
            try:
                 data = read_data(dir_file_path)
            except Exception as e:
                print(dir_file)
                shutil.move(dir_file_path,error_file_path)

            # data = data[:,:,25]
            data = np.array(data)
            #data.resize((109, 109, 109), refcheck=False)
            #dataPath_list.append(dir_file_path)
            dataImg_list.append(data)

            #dataPath = os.path.dirname(dir_file_path)
            #dataPath = dataPath.split('/')[-1]
            #data_label.append(dataPath)

            #dir_file = dir_file.split('_')[-1]
            data_label.append(dir_file.split('_')[0])

    return dataImg_list,data_label


dataPath_list=[]
data_label=[]
dataImg_list=[]

root_path = './mapdcm2nii'
error_path='./error'

dataImg_list,data_label=get_file_bypath(root_path)
