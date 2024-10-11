import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#from AttResUnet import Attention_ResUNet
#from Model.seg_hrnet import seg_hrnet
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight
# from keras.utils.np_utils import *
from tensorflow.keras import utils
import numpy as np 
import os
import keras_tuner as kt
from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.models import Model
# # from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
# from keras.layers import merge

from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
#from hyperopt import hp
from keras_tuner import HyperParameters, Hyperband, RandomSearch
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import backend as K  
#K.set_image_data_format('channels_first')  
#K.set_image_data_format('channels_last') 
import numpy as np
import pandas as pd
from tensorflow.python.keras.utils import np_utils
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
#from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Conv2D,MaxPooling2D,UpSampling2D,Dropout,Flatten 
from tensorflow.keras.layers import BatchNormalization,AveragePooling2D,concatenate  
from tensorflow.keras.layers import ZeroPadding2D,add
from tensorflow.keras.layers import Dropout, Activation
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.models import Model,load_model
#from tensorflow.keras.utils.np_utils import to_binary
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers, regularizers # 优化器，正则化项
from tensorflow.keras.optimizers import SGD, Adam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn import preprocessing


# df1 = pd.read_csv(r"E:/huan/ml/cnn/new/i341_j368.csv")
# df2 = pd.read_csv(r"E:/huan/ml/cnn/new/i580_j316.csv")
df1 = pd.read_csv(r"E:\浣雨柯-数据与代码\数据\卷积神经网络滑坡易发性分析\new\i341_j368.csv")
df2 = pd.read_csv(r"E:\浣雨柯-数据与代码\数据\卷积神经网络滑坡易发性分析\new\i580_j316.csv")
# df3 = pd.read_csv(r"F:/huan/ml/data/sudataset_clip3.csv")
data1 = df1.fillna(0)
data2 = df2.fillna(0)
#data3 = df3.fillna(0)
minmax = preprocessing.MinMaxScaler()
data1 = minmax.fit_transform(data1[['rivers','roads','aspect','plane','twi','soil','landuse','altitude','relief','roughness','rainfall','slope','ndvi','profile','lithology']])
data2 = minmax.fit_transform(data2[['rivers','roads','aspect','plane','twi','soil','landuse','altitude','relief','roughness','rainfall','slope','ndvi','profile','lithology']])
#data3 = minmax.fit_transform(data3[['Rivers','roads','ndvi','aspect','plane','profile','twi','soiltype','landuse','altitude','relief','roughness','rainfall','slope']])
print(data1)
print(data2)
#print(data3)
#X = np.expand_dims(df.values[:, 1:-1].astype(float), axis=2)
x1_data1 = np.expand_dims(data1.astype(float), axis=2)
y1_data1 = df1.values[:, -2]
print(y1_data1)

print(1)
I=0
J=0
K=0
i=0
j=0
k=0
X_data=[]
Y_data=[]
while J<325:
    i=0
    while i<352:
        j=0
        while j<17:
            X_data.append(x1_data1[i+368*J+j*368:(i+368*J+j*368)+17,:])
            j=j+1
        i=i+1
    J=J+1
                                
x_data=np.array(X_data)
x_data=x_data.reshape(-1,15)
x_data=x_data.reshape(-1,15,17,17)
I=0
J=0
K=0
i=0
j=0
k=0
while J<325:
    i=0
    while i<352:
        Y_data.append(y1_data1[i+8+8*368+J*368])
        i=i+1
    J=J+1

x1_data2 = np.expand_dims(data2.astype(float), axis=2)
y1_data2 = df2.values[:, -2]
print(1)
I=0
J=0
K=0
i=0
j=0
k=0
while J<564:
    i=0
    while i<300:
        j=0
        while j<17:
            X_data.append(x1_data2[i+316*J+j*316:(i+316*J+j*316)+17,:])
            j=j+1
        i=i+1
    J=J+1
                                

x_data=np.array(X_data)
x_data=x_data.reshape(-1,15)
x_data=x_data.reshape(-1,15,17,17)
I=0
J=0
K=0
i=0
j=0
k=0
while J<564:
    i=0
    while i<300:
        Y_data.append(y1_data2[i+8+8*316+J*316])
        i=i+1
    J=J+1
    
# x1_data3 = np.expand_dims(data3.astype(float), axis=2)
# y1_data3 = df3.values[:, -2]
# print(1)
# I=0
# J=0
# K=0
# i=0
# j=0
# k=0
# while J<215:
#     i=0
#     while i<240:
#         j=0
#         while j<17:
#             X_data.append(x1_data3[i+256*J+j*256:(i+256*J+j*256)+17,:])
#             j=j+1
#         i=i+1
#     J=J+1
                                
# x_data=np.array(X_data,dtype='float32')
# x_data=x_data.reshape(-1,14)
# x_data=x_data.reshape(-1,14,17,17)
# I=0
# J=0
# K=0
# i=0
# j=0
# k=0
# while J<215:
#     i=0
#     while i<240:
#         Y_data.append(y1_data3[i+8+8*256+J*256])
#         i=i+1
#     J=J+1
    
    
    
x2_data1 = np.expand_dims(data1.astype(float), axis=2)
y2_data1 = df1.values[:, -2]

print(1)
I=0
J=0
K=0
i=0
j=0
k=0
X_data=[]
Y_data=[]
while J<331:
    i=0
    while i<358:
        j=0
        while j<11:
            X_data.append(x1_data1[i+368*J+j*368:(i+368*J+j*368)+11,:])
            j=j+1
        i=i+1
    J=J+1
                                
x_data=np.array(X_data)
x_data=x_data.reshape(-1,15)
x_data=x_data.reshape(-1,15,11,11)
I=0
J=0
K=0
i=0
j=0
k=0
while J<331:
    i=0
    while i<358:
        Y_data.append(y1_data1[i+5+5*368+J*368])
        i=i+1
    J=J+1

x2_data2 = np.expand_dims(data2.astype(float), axis=2)
y2_data2 = df2.values[:, -2]
print(1)
I=0
J=0
K=0
i=0
j=0
k=0
while J<570:
    i=0
    while i<306:
        j=0
        while j<11:
            X_data.append(x1_data2[i+316*J+j*316:(i+316*J+j*316)+11,:])
            j=j+1
        i=i+1
    J=J+1
                                

x_data=np.array(X_data)
x_data=x_data.reshape(-1,15)
x_data=x_data.reshape(-1,15,11,11)
I=0
J=0
K=0
i=0
j=0
k=0
while J<570:
    i=0
    while i<306:
        Y_data.append(y1_data2[i+5+5*316+J*316])
        i=i+1
    J=J+1
    
# x2_data3 = np.expand_dims(data3.astype(float), axis=2)
# y2_data3 = df3.values[:, -2]
# print(1)
# I=0
# J=0
# K=0
# i=0
# j=0
# k=0
# while J<221:
#     i=0
#     while i<246:
#         j=0
#         while j<11:
#             X_data.append(x2_data3[i+256*J+j*256:(i+256*J+j*256)+11,:])
#             j=j+1
#         i=i+1
#     J=J+1
                                
# x_data=np.array(X_data,dtype='float32')
# x_data=x_data.reshape(-1,14)
# x_data=x_data.reshape(-1,14,11,11)
# I=0
# J=0
# K=0
# i=0
# j=0
# k=0
# while J<221:
#     i=0
#     while i<246:
#         Y_data.append(y2_data3[i+5+5*256+J*256])
#         i=i+1
#     J=J+1
    
x3_data1 = np.expand_dims(data1.astype(float), axis=2)
y3_data1 = df1.values[:, -2]

print(1)
I=0
J=0
K=0
i=0
j=0
k=0
X_data=[]
Y_data=[]
while J<337:
    i=0
    while i<364:
        j=0
        while j<5:
            X_data.append(x1_data1[i+368*J+j*368:(i+368*J+j*368)+5,:])
            j=j+1
        i=i+1
    J=J+1
                                
x_data=np.array(X_data)
x_data=x_data.reshape(-1,15)
x_data=x_data.reshape(-1,15,5,5)
I=0
J=0
K=0
i=0
j=0
k=0
while J<337:
    i=0
    while i<364:
        Y_data.append(y1_data1[i+2+2*368+J*368])
        i=i+1
    J=J+1

x3_data2 = np.expand_dims(data2.astype(float), axis=2)
y3_data2 = df2.values[:, -2]
print(1)
I=0
J=0
K=0
i=0
j=0
k=0
while J<576:
    i=0
    while i<312:
        j=0
        while j<5:
            X_data.append(x1_data2[i+316*J+j*316:(i+316*J+j*316)+5,:])
            j=j+1
        i=i+1
    J=J+1
                                

x_data=np.array(X_data)
x_data=x_data.reshape(-1,15)
x_data=x_data.reshape(-1,15,5,5)
I=0
J=0
K=0
i=0
j=0
k=0
while J<576:
    i=0
    while i<312:
        Y_data.append(y1_data2[i+2+2*316+J*316])
        i=i+1
    J=J+1
    
    
# x3_data3 = np.expand_dims(data3.astype(float), axis=2)
# y3_data3 = df3.values[:, -2]
# print(1)
# I=0
# J=0
# K=0
# i=0
# j=0
# k=0
# while J<227:
#     i=0
#     while i<252:
#         j=0
#         while j<5:
#             X_data.append(x3_data3[i+256*J+j*256:(i+256*J+j*256)+5,:])
#             j=j+1
#         i=i+1
#     J=J+1
                                
# x_data=np.array(X_data,dtype='float32')
# x_data=x_data.reshape(-1,14)
# x_data=x_data.reshape(-1,14,5,5)
# I=0
# J=0
# K=0
# i=0
# j=0
# k=0
# while J<227:
#     i=0
#     while i<252:
#         Y_data.append(y3_data3[i+2+2*256+J*256])
#         i=i+1
#     J=J+1
    


    
y_data=np.array(Y_data)
y_data=y_data.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size =0.2, random_state=30)
X_test=X_test.reshape(-1,15,5,5)
X_train=X_train.reshape(-1,15,5,5)

y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)
y_data = np_utils.to_categorical(y_data, num_classes=2)
y_data=np.array(y_data,dtype=np.int8)

counts = np.bincount(y_data[:, 0])
# 基于数量计算类别权重
weight_for_0 = 8. / counts[1]
weight_for_1 = 9. / counts[0]
class_weight = {0: weight_for_0, 1: weight_for_1}
print (class_weight)

def focal_loss(alpha=0.5, gamma=1.5, epsilon=1e-6):
    print('*'*20, 'alpha={}, gamma={}'.format(alpha, gamma))
    def focal_loss_calc(y_true, y_probs):
        positive_pt = tf.where(tf.equal(y_true, 1), y_probs, tf.ones_like(y_probs))
        negative_pt = tf.where(tf.equal(y_true, 0), 1-y_probs, tf.ones_like(y_probs))
        
        loss =  -alpha * tf.pow(1-positive_pt, gamma) * tf.math.log(tf.clip_by_value(positive_pt, epsilon, 1.)) - \
            (1-alpha) * tf.pow(1-negative_pt, gamma) * tf.math.log(tf.clip_by_value(negative_pt,  epsilon, 1.))

        return tf.reduce_sum(loss)
    return focal_loss_calc




print(1)

model = Sequential()
model.add(Conv2D(
        32,
        3, # depth
		padding = 'same',
		input_shape=(15,5,5)
						)
		)
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(MaxPooling2D(
					pool_size = (2,2),
					strides = (2,2),
					padding = 'same',
					)
		)

##2:128
model.add(Conv2D(128, 3,padding = 'same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling2D(2, 2, padding = 'same'))

##3:256
model.add(Conv2D(256,3,padding = 'same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling2D(2, 2, padding = 'same'))

##4:512
model.add(Conv2D(512, 3,padding = 'same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling2D(2, 2,padding = 'same'))

##5:512
model.add(Conv2D(512, 3, padding = 'same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling2D(2, 2, padding = 'same'))

#####FC
model.add(Flatten())
#model.add(BatchNormalization())
model.add(Dense(4096, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
hp_units = hp.Int('units', min_value=25, max_value=200, step=25)
model.add(Dense(hp_units, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))
########################

hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
hp_optimizer=hp.Choice('Optimizer', values=['Adam', 'SGD'])

if hp_optimizer == 'Adam':
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
elif hp_optimizer == 'SGD':
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    nesterov=True
    momentum=0.9

model.compile(loss = 'binary_crossentropy',optimizer = hp_optimizer,metrics=['accuracy'])  
model.summary()  


hp = HyperParameters()
#Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4]))
#hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4])
#loss = focal(alpha = hp.Float('alpha', min_value=0,max_value=1), gamma = hp.Float('gamma', min_value=0,max_value=5))
# model.compile(optimizer = hp_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#  回调函数
#  val_loss连续10轮没有下降则停止训练
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
# tuner_mlp = kt.tuners.BayesianOptimization(model, objective='val_loss', max_trials=30, directory='.', project_name='tuning-mlp')
#tuner_mlp.search(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), callbacks=early_stopping)

tuner = RandomSearch(
    build_model,
    # `tune_new_entries=False` prevents unlisted parameters from being tuned
    objective='val_accuracy',
    max_trials=30,
    directory='my_dir50',
    project_name='50')

# tuner.search(x_train, y_train[:,:-1,],
#              validation_data=(x_test, y_test))
tuner.search(train_Generator,
            steps_per_epoch = steps_per_epoch,
            epochs = epochs,
            callbacks = [early_stopping],
            validation_data = validation_data,
            validation_steps = validation_steps)
tuner.search_space_summary()
models = tuner.get_best_models(num_models=2)
tuner.results_summary()
"""
Keras Tuner 提供了四种调节器：
RandomSearch、Hyperband、BayesianOptimization和Sklearn
"""
# #实例化调节器并执行超调
# tuner = kt.Hyperband(model,
#                      objective='val_accuracy',
#                      max_epochs=10,
#                      factor=3,
#                      directory='my_dir',
#                      project_name='intro_to_kt')


# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]


#训练模型
"""
使用从搜索中获得的超参数找到训练模型的最佳生命周期数。
"""
# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

#最佳循环数不再进行训练
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hyper = model.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2)

#测试数据上评估超模型
eval_result = hypermodel.evaluate(X_test, y_test)
print("[test loss, test accuracy]:", eval_result)