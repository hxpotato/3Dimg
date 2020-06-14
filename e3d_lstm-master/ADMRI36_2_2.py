#训练测试分开，3D卷积-2class--32*32*32-run
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
batch_size = 10
learning_rate =0.01
dropout_rate = 0.1
epoch_num = 150


def VGG16():
     model = models.Sequential()
     model.add(Conv3D(32, (3,3,3), activation='relu', padding='same', input_shape=(32, 32, 32,1), kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(MaxPooling3D(pool_size=[2,2,2],strides=None))

     model.add(Conv3D(64, (3,3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(MaxPooling3D(pool_size=[2,2,2],strides=None))


     model.add(Conv3D(128, (3,3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(Conv3D(128, (3,3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(MaxPooling3D(pool_size=[2,2,2],strides=None))

     model.add(Conv3D(256, (3,3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(Conv3D(256, (3,3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(MaxPooling3D(pool_size=[2,2,2],strides=None))


     model.add(Conv3D(256, (3,3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(Conv3D(256, (3,3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
     model.add(MaxPooling3D(pool_size=[2,2,2],strides=None))
#
     model.add(Flatten())  # 2*2*512
     model.add(Dense(4068, activation='relu'))
    # model.add(Dropout(0.5))
     model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.5))
     model.add(Dense(10, activation='relu'))
    # model.add(Dropout(0.5))
     model.add(Dense(2, activation='softmax'))

     return model

def scheduler(epoch):
    if epoch < epoch_num * 0.4:
        return learning_rate
    if epoch < epoch_num * 0.8:
        return learning_rate * 0.1
    return learning_rate * 0.01

#tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE
MRIAD=pd.read_csv('MRI_Longitudinal_NoNANWithReduceLabel_900s_ADvsMCI_scalar_fill.csv')

MRIAD_data=MRIAD.iloc[:,9:MRIAD.columns.size]
MRIAD_data_jion=np.zeros((len(MRIAD_data),32407),dtype=np.float64)
MRIAD_data_jion=pd.DataFrame(MRIAD_data_jion)
MRIAD_data=pd.concat([MRIAD_data,MRIAD_data_jion],axis=1,join='inner')

from sklearn.utils import shuffle
MRIAD=shuffle(MRIAD)

print(MRIAD.shape)
print(len(MRIAD))
print(MRIAD.columns.size)

#MRIAD_data=MRIAD.iloc[:,9:MRIAD.columns.size]
MRIAD_label=MRIAD.iloc[:,4:5]
MRIAD_label=MRIAD_label['DX'].apply(lambda x: switch(x)).values

#from sklearn.utils import shuffle
#MRIAD=shuffle(MRIAD)


data_size = len(MRIAD_data)
train_test_split = (int)(data_size * 0.2)

#MRIAD_data_jion=pd.DataFrame(np.zeros(len(MRIAD_data),32407))
#MRIAD_data_jion=pd.DataFrame(np.zeros((len(MRIAD_data),32407),dtype=np.float64))
#MRIAD_data_jion=np.zeros((len(MRIAD_data),32407),dtype=np.float64)
#MRIAD_data_jion=pd.DataFrame(MRIAD_data_jion)
#MRIAD_data_jion=np.zeros((len(MRIAD_data),32407),dtype=np.float64)
x_train_join=MRIAD_data_jion[train_test_split:]
x_test_join=MRIAD_data_jion[:train_test_split]


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


#x_train=tf.expand_dims(x_train,axis=4)
#x_test=tf.expand_dims(x_test,axis=4)

print(x_train.shape)

#if __name__ == '__main__':
model = VGG16()
model.summary()
sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
#adam=tf.train.AdamOptimizer(lr=learning_rate)
change_lr = LearningRateScheduler(scheduler)
import time
from tensorflow.keras.callbacks import TensorBoard
#model_name = "mode-{}".format(int(time.time()))
#tensorboard = TensorBoard(log_dir='vgg16_from_tensorflow20/{}'.format(model_name))
data_augmentation = False
#tf.keras.backend.set_learning_phase(True)

if not data_augmentation:
        print('Not using data augmentation')
        #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        #model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

       # checkpoint_filepath = 'models/model-ep{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5'
       # checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

        model.summary()

        #metrics = Metrics()


        #model.fit(x_train,y_train,batch_size=batch_size,steps_per_epoch=20,epochs=epoch_num,callbacks=[change_lr],validation_data=(x_test,y_test))
       # model.fit(x_train, y_train, batch_size=batch_size, steps_per_epoch=2, epochs=200, callbacks=[change_lr],validation_split=0.2,validation_data=(x_test, y_test))
       # history=model.fit(x_train, y_train, batch_size=batch_size, steps_per_epoch=1,epochs=epoch_num, callbacks=[change_lr])
#        history = LossHistory()
        history=model.fit(x_train,y_train, batch_size=batch_size, steps_per_epoch=2,epochs=epoch_num,callbacks=[change_lr])

       # model.fit(x_train, y_train, batch_size=batch_size, steps_per_epoch=2, epochs=200,validation_data=(x_test, y_test))
else:
        print('Using real-time data augmentation')
      #  train_datagen= keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,rotation_range=10,width_shift_range=0.1,height_shift_range=0.1,
       # train_datagen = keras.preprocessing.image.ImageDataGenerator()                                                          # shear_range=0.1,zoom_range=0.1,horizontal_flip=False,fill_mode='nearest')


       # test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
       # test_datagen = keras.preprocessing.image.ImageDataGenerator()

        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
       # model.fit(train_datagen.flow(x_train,y_train, batch_size=32),steps_per_epoch=x_train.shape[0],epochs=200, callbacks=[change_lr],validation_data=test_datagen.flow(x_test,y_test, batch_size=32),validation_steps=x_test.shape[0])
        model.fit(x_train,y_train,batch_size=32,validation_split=0.2,steps_per_epoch=x_train.shape[0],epochs=200)

validation_steps = 2
loss0, accuracy0 = model.evaluate(x_test,y_test, steps=validation_steps)
print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))
plt.plot(accuracy0)
plt.plot(loss0)
plt.plot(history.history['loss'])


plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

