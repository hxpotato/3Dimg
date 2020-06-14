#训练测试分开，3D卷积
import pandas as pd
import numpy as np
import tensorflow as tf
#import tensorflow.keras as keras
from tensorflow.keras import models,optimizers,regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Conv3D,MaxPooling3D, Activation
from tensorflow import keras
import matplotlib.pyplot as plt
#from keras.utils import np_utils
#from tensorflow.python.layers.pooling import
from tensorflow.keras.callbacks import Callback,ModelCheckpoint
#from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential


import pickle as p
import os

label_names={0: 1, 2: 2, 1: 3, 2 : 4 , 1 : 5, 1 : 6, 0 : 7 , 2 : 8}
label_key=[0,1,2]

def switch(arg):
     switcher = {
         1: 0,  # stable NL
         2: 2,  # stable MCI
         3: 1,  # stable AD
         4: 2,  # to MCI
         5: 1,  # to AD
         6: 1,  # to AD
         7: 0,  # to NL
         8: 2,  # to MCI
    }
     return switcher.get(arg, -1)
#
#

weight_decay = 5e-4
batch_size = 16
learning_rate =0.0001
dropout_rate = 0.8
epoch_num = 150


def CNN():
     model = models.Sequential()
     model.add(Conv3D(50, (5,5,5), activation='relu', strides=1, padding='same', input_shape=(32, 32, 32,1), kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(Conv3D(50, (5,5,5), activation='relu', strides=2, padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
     #model.add(MaxPooling3D(pool_size=[1, 1, 1], strides=None))

     model.add(Conv3D(100, (3,3,3), activation='relu',strides=1, padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(Conv3D(200, (3,3,3), activation='relu',strides=2, padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(Conv3D(400, (3,3,3), activation='relu', strides=1, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(Conv3D(600, (3,3,3), activation='relu', strides=2, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(Conv3D(800, (3,3,3), activation='relu', strides=1, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(Conv3D(1000, (3,3,3), activation='relu', strides=2, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(Conv3D(1200, (3,3,3), activation='relu', strides=1, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(Conv3D(1400, (3,3,3), activation='relu', strides=2, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(Conv3D(1500, (3,3,3), activation='relu', strides=1, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(Conv3D(1600, (3,3,3), activation='relu', strides=2, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
     #model.add(MaxPooling3D(pool_size=[1,1,1],strides=None))

#
     model.add(Flatten())  # 2*2*512
     model.add(Dense(1024, activation='relu'))
     model.add(Dense(3, activation='softmax'))

     return model

def scheduler(epoch):
    if epoch < epoch_num * 0.4:
        return learning_rate
    if epoch < epoch_num * 0.8:
        return learning_rate * 0.1
    return learning_rate * 0.01

#tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE
MRIAD=pd.read_csv('MRI_Longitudinal_NoNANWithReduceLabel_900s_ADvsNCvsMCI_scalar_fill.csv')

MRIAD_data=MRIAD.iloc[:,9:MRIAD.columns.size]
MRIAD_data_jion=np.zeros((len(MRIAD_data),32407),dtype=np.float64)
MRIAD_data_jion=pd.DataFrame(MRIAD_data_jion)
MRIAD_data=pd.concat([MRIAD_data,MRIAD_data_jion],axis=1,join='inner')

from sklearn.utils import shuffle
MRIAD=shuffle(MRIAD)

print(MRIAD.shape)
print(len(MRIAD))
print(MRIAD.columns.size)


#MRIAD_label=MRIAD.iloc[:,4:5]
MRIAD_label=MRIAD.iloc[:,4:5]
#MRIAD_label['DX']=MRIAD_label['DX'].apply(lambda x: switch(x)).values
MRIAD_label=MRIAD_label['DX'].apply(lambda x: switch(x)).values

data_size = len(MRIAD_data)
train_test_split = (int)(data_size * 0.2)

#MRIAD_data_jion=pd.DataFrame(np.zeros((len(MRIAD_data),32407)))

#x_train_join=MRIAD_data_jion[train_test_split:]
#x_test_join=MRIAD_data_jion[:train_test_split]

x_train=MRIAD_data[train_test_split:]
x_test=MRIAD_data[:train_test_split]

y_train= MRIAD_label[train_test_split:]
y_train = pd.get_dummies(y_train)
y_test = MRIAD_label[:train_test_split]
y_test = pd.get_dummies(y_test)

y_train=y_train.to_numpy()
y_test=y_test.to_numpy()

#x_train=pd.concat([x_train,x_train_join],axis=1,join='inner')
#x_test=pd.concat([x_test,x_test_join],axis=1,join='inner')

x_train=x_train.values.reshape(x_train.shape[0],32,32,32,1)
x_test=x_test.values.reshape(x_test.shape[0],32,32,32,1)

#if __name__ == '__main__':
model = CNN()
model.summary()
sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
#adam=tf.train.AdamOptimizer(lr=learning_rate)
change_lr = LearningRateScheduler(scheduler)
data_augmentation = False


if not data_augmentation:
        print('Not using data augmentation')
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        model.summary()
        history=model.fit(x_train,y_train, batch_size=batch_size,steps_per_epoch=2,epochs=epoch_num,callbacks=[change_lr])
else:
        print('Using real-time data augmentation')
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])      # model.fit(train_datagen.flow(x_train,y_train, batch_size=32),steps_per_epoch=x_train.shape[0],epochs=200, callbacks=[change_lr],validation_data=test_datagen.flow(x_test,y_test, batch_size=32),validation_steps=x_test.shape[0])
        model.fit(x_train,y_train,batch_size=32,validation_split=0.2,steps_per_epoch=x_train.shape[0],epochs=10)

validation_steps = 20
loss0, accuracy0 = model.evaluate(x_test,y_test, steps=validation_steps)
print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))
plt.plot(accuracy0)
plt.plot(loss0)

#plt.plot(history.history)

#plt.plot(history.history['acc'])
plt.plot(history.history['loss'])


plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

