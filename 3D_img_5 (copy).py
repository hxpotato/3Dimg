#from __future__ import print_function
#import torch
import sklearn as sk
import copy
import numpy as np
import pandas as pd
import os
import shutil
import nibabel as nib
import threading
#import tensorlayer as tl
import matplotlib.pyplot as plt

import scipy
#from tensorlayer.prepro import *
import skimage.measure
import sklearn as sk
import sklearn.model_selection

from  skimage.transform import resize
#from functools import reduce
#from utils import nice_print, mem_report, cpu_stats
#import copy
#import operator                      c
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
from E3DLSTM_tf import rnn

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

            data_ID.append(dir_file.split('_')[-1].split('.')[0])
            data_label.append(dir_file.split('_')[0])
            global data_list
            data_list=data_list.append({'ID':dir_file.split('_')[-1].split('.')[0],
                                'ImagePath':dir_file_path,
                                #'Img':data,
                                'label':dir_file.split('_')[0],
                                'Sex':dir_file.split('_')[1],
                                'Age':dir_file.split('_')[2],
                                'Visit':dir_file.split('_')[3],
                                'Subject':dir_file.split('_')[6]+"_"+dir_file.split('_')[7]+"_"+dir_file.split('_')[8]},ignore_index=True)
           # data_list=data_list.append(item,index='data_ID')
    print(data_list.head())
             #'Image':data,
            #data_list
           # data_list.append(data_ID,dataImg_list,data_label
    return data_list,data_ID,dataImg_list,data_label
def fetch_data(root_path):
        data_list,data_ID,dataImg_list,data_label=get_file_bypath(root_path)
         #data_list_filter=data_list[data_list['data_label'].isin(['AD','CN'])]
        data_list_filter=data_list[data_list['label'].isin(['AD','MCI','CN'])]
          #data_list_filter=data_list[data_list['data_label'].isin(['AD','LMCI','SMCI','EMCI','CN'])]
        #data_list_filter.sort_values(by=['Subject','Visit'],axis=1,ascending=True)
        data_group_count= data_list_filter.groupby('Subject').count()
        #print(data_group_count['Visit'])
        #if data_group_count['Visit']>1:
           #Subject1= data_list_filter[data_group_count['Visit']>1].sort_values(by=['Subjecct'], ascending=True)
        Subject1= data_group_count[data_group_count['Visit']>1].index
        print(Subject1)
        #else:
        Subject2= data_group_count[data_group_count['Visit']==1].index
        print(Subject2)
        #Subjects = data_list_filter.drop_duplicates(subset='Subject')
        #for i in Subjects['Subject']:
        for i in Subject1:
                Subject= data_list_filter[data_list_filter['Subject']==i].sort_values(by=['Visit'], ascending=True)
                for j in Subject['Visit']:
                   Subject_img_path= Subject[Subject['Visit']==j]
                  # data=read_data(str(Subject_img_path['ImagePath'].values))
                   Img_path =str(Subject_img_path['ImagePath'].values)

                   data=read_data(Img_path[2:-2])
                   data=resize(data,(256,256,256))
                   #Img= np.zeros(256,256,256)
                   #size=data.size()
                  # Img=copy.deepcopy(data)
                   #np.array(data)
                   Subject_img_list1.append(data)

        return Subject_img_list1

dataPath_list=[]
data_label=[]
data_ID=[]
dataImg_list=[]
data_list = pd.DataFrame(columns=('ID','ImagePath','label','Sex','Age','Visit','Subject'))
#data_ID_Idx=[]
Subject_img_list=[]
#root_path = './mapdcm2nii'
root_path = './test'
error_path='./error'

Subject_img_list1=[]
Subject_img_list2=[]
#data_list_group = data_list_filter.groupby('Subject').apply(lambda x:x.sort_values('Visit',ascending=True)).reset_index(drop=True)
#data_list_group = data_list_filter.groupby('Subject')


#data_list_group.reset_index(drop=True)

#for st recent call last):

 #   print(test(['Subject','Visit','data_label']))
fetch_data(root_path)

print('ok')

