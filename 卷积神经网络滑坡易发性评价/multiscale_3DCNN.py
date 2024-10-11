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


def plot_confusion_matrix1(cm, classes,
    title='Confusion matrix',
    cmap=plt.cm.Greens):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.savefig('E:/huan/ml/cnn/new/figure/cm50_32_17_56.png')
    plt.savefig(r'E:\CNN-res\cm50_32_17_56.png')
    plt.show()

# 显示混淆矩阵
def plot_confuse1(model, x_val, y_val):
    predictions = model.predict_classes(x_val)
    #predictions=np.argmax(predictions,axis=-1)
    truelabel = y_val.argmax(axis=-1) # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure(figsize=(10,10))
    plot_confusion_matrix1(conf_mat, range(np.max(truelabel)))


def focal_loss(alpha=0.5, gamma=1.5, epsilon=1e-6):
    print('*' * 20, 'alpha={}, gamma={}'.format(alpha, gamma))

    def focal_loss_calc(y_true, y_probs):
        positive_pt = tf.where(tf.equal(y_true, 1), y_probs, tf.ones_like(y_probs))
        negative_pt = tf.where(tf.equal(y_true, 0), 1 - y_probs, tf.ones_like(y_probs))

        loss = -alpha * tf.pow(1 - positive_pt, gamma) * tf.math.log(tf.clip_by_value(positive_pt, epsilon, 1.)) - \
               (1 - alpha) * tf.pow(1 - negative_pt, gamma) * tf.math.log(tf.clip_by_value(negative_pt, epsilon, 1.))

        return tf.reduce_sum(loss)

    return focal_loss_calc
'''
step1 从csv读取数据
df1和df2是读取的csv构成的datarrame变量
'''
# df1 = pd.read_csv(r"E:/huan/ml/cnn/new/i341_j368.csv")
# df2 = pd.read_csv(r"E:/huan/ml/cnn/new/i580_j316.csv")
""
df1 = pd.read_csv(r"F:\Tangjiacheng-CSU\CNN4LSM\data\卷积神经网络滑坡易发性分析\new\i341_j368.csv")
df2 = pd.read_csv(r"F:\Tangjiacheng-CSU\CNN4LSM\data\卷积神经网络滑坡易发性分析\new\i580_j316.csv")

# df3 = pd.read_csv(r"F:/huan/ml/data/sudataset_clip3.csv")
'''
step2 处理缺失值 
data1和data2是补充缺失值的标签
'''
data1 = df1.fillna(0)
data2 = df2.fillna(0)
#data3 = df3.fillna(0)
'''
step3 特征归一化
将滑坡影响因子进行特征归一化
'''
minmax = preprocessing.MinMaxScaler()
data1 = minmax.fit_transform(data1[['rivers','roads','aspect','plane','twi','soil','landuse','altitude','relief','roughness','rainfall','slope','ndvi','profile','lithology']])
data2 = minmax.fit_transform(data2[['rivers','roads','aspect','plane','twi','soil','landuse','altitude','relief','roughness','rainfall','slope','ndvi','profile','lithology']])
#data3 = minmax.fit_transform(data3[['Rivers','roads','ndvi','aspect','plane','profile','twi','soiltype','landuse','altitude','relief','roughness','rainfall','slope']])
print(data1)
print(data2)
#print(data3)
#X = np.expand_dims(df.values[:, 1:-1].astype(float), axis=2)
'''
step4 
x1_data1获取成三维格式 样本数*特征数*通道
y1_data1为真实标签数组
'''
x1_data1 = np.expand_dims(data1.astype(float), axis=2)
# csv的remote代表标签
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
'''
'''
while J<325:
    i=0
    while i<352:
        j=0
        while j<17:
            X_data.append(x1_data1[i+368*J+j*368:(i+368*J+j*368)+17,:])
            j=j+1
        i=i+1
    J=J+1
'''
step5
第一个循环
'''
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

from collections import Counter
# from imblearn.over_sampling import RandomOverSampler, SMOTE
print(x_data)
print(y_data)
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size =0.2, random_state=30)
X_test=X_test.reshape(-1,15,5,5)
X_train=X_train.reshape(-1,15,5,5)

y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)
y_data = np_utils.to_categorical(y_data, num_classes=2)
y_data=np.array(y_data,dtype=np.int8)

# smote = SMOTE(random_state=0)  # random_state为0（此数字没有特殊含义，可以换成其他数字）使得每次代码运行的结果保持一致
# X_smotesampled, y_smotesampled = smote.fit_resample(X_train, y_train)  # 使用原始数据的特征变量和目标变量生成过采样数据集
# smote = SMOTE(random_state=0)  # random_state为0（此数字没有特殊含义，可以换成其他数字）使得每次代码运行的结果保持一致
# X_smotesampled, y_smotesampled = smote.fit_resample(X_test, y_test)  # 使用原始数据的特征变量和目标变量生成过采样数据集

counts = np.bincount(y_data[:, 0])
# 基于数量计算类别权重
weight_for_0 = 8. / counts[1]
weight_for_1 = 9. / counts[0]
class_weight = {0: weight_for_0, 1: weight_for_1}
print (class_weight)

# def focal_loss(alpha=0.5, gamma=1.5, epsilon=1e-6):
#     print('*'*20, 'alpha={}, gamma={}'.format(alpha, gamma))
#     def focal_loss_calc(y_true, y_probs):
#         positive_pt = tf.where(tf.equal(y_true, 1), y_probs, tf.ones_like(y_probs))
#         negative_pt = tf.where(tf.equal(y_true, 0), 1-y_probs, tf.ones_like(y_probs))
#
#         loss =  -alpha * tf.pow(1-positive_pt, gamma) * tf.math.log(tf.clip_by_value(positive_pt, epsilon, 1.)) - \
#             (1-alpha) * tf.pow(1-negative_pt, gamma) * tf.math.log(tf.clip_by_value(negative_pt,  epsilon, 1.))
#
#         return tf.reduce_sum(loss)
#     return focal_loss_calc


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
model.add(Dense(128, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))
########################
adam = Adam(lr = 1e-4)

model.compile(loss = 'binary_crossentropy',optimizer=adam,metrics=['accuracy'])  
model.summary()  

history=model.fit(
    x_data, y_data,
    #steps_per_epoch=nb_train_samples // batch_size,
    epochs=25, batch_size=64,
    validation_data=(x_data, y_data),
    class_weight=class_weight
    #validation_steps=nb_validation_samples // batch_size)
    )
model.save(r'E:\CNN-res\50_32_17_56.h5')   # HDF5文件，pip install h5py
# //model.save('E:/huan/ml/cnn/new/model/50_32_17_56.h5')
print('\nSuccessfully saved as a model')

# def plot_confusion_matrix1(cm, classes,
#     title='Confusion matrix',
#     cmap=plt.cm.Greens):
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     # plt.savefig('E:/huan/ml/cnn/new/figure/cm50_32_17_56.png')
#     plt.savefig(r'E:\CNN-res\cm50_32_17_56.png')
#     plt.show()

# # 显示混淆矩阵
# def plot_confuse1(model, x_val, y_val):
#     predictions = model.predict_classes(x_val)
#     #predictions=np.argmax(predictions,axis=-1)
#     truelabel = y_val.argmax(axis=-1) # 将one-hot转化为label
#     conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
#     plt.figure(figsize=(10,10))
#     plot_confusion_matrix1(conf_mat, range(np.max(truelabel)))

train_history = load_model(r'E:\CNN-res\50_32_17_56.h5',custom_objects={'focal_loss_calc': focal_loss})
# plot_confuse1(train_history,x_data, y_data)

# plot training accuracy and loss from history
plt.figure(figsize=(12,9))
#plt.subplot(121)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy',fontsize=16)
plt.ylabel('accuracy',fontsize=14)
plt.xlabel('epoch',fontsize=14)
plt.legend(['train', 'test'],fontsize=16,loc='upper left')
# plt.savefig('E:/huan/ml/cnn/new/figure/acc50_32_17_56.png')

plt.savefig(r'E:\CNN-res\acc50_32_17_56.png')
plt.figure(figsize=(12,9))
#plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss',fontsize=16)
plt.ylabel('loss',fontsize=14)
plt.xlabel('epoch',fontsize=14)
plt.legend(['train', 'test'],fontsize=16, loc='upper right')
# plt.savefig('E:/huan/ml/cnn/new/figure/loss50_32_17_56.png')
plt.savefig(r'E:\CNN-res\loss50_32_17_56.png')
plt.show()

# ROC curves
from sklearn import metrics
import pylab as plt

Font={'size':18, 'family':'Times New Roman'}

# y_probas1 = model.predict_proba(X_test)
# y_scores1 = y_probas1[:,1]
# truelabel = y_test.argmax(axis=-1)
# 使用 predict 方法获取预测概率
y_scores1 = model.predict(X_test)[:, 1]
# 将 one-hot 编码的标签转换为 1D 数组
truelabel = y_test.argmax(axis=-1)

fpr1,tpr1,thres1 = metrics.roc_curve(truelabel, y_scores1,drop_intermediate=False)
roc_auc1 = metrics.auc(fpr1, tpr1)
print(roc_auc1)
plt.figure(figsize=(6,6))
plt.plot(fpr1, tpr1, 'b', label = 'AUC = %0.4f' % roc_auc1, color='Blue')
plt.legend(loc = 'lower right', prop=Font)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate', Font)
plt.xlabel('False Positive Rate', Font)
plt.tick_params(labelsize=15)
# plt.savefig('E:/huan/ml/cnn/new/figure/roc50_32_17_56.png')
plt.savefig(r'E:\CNN-res\roc50_32_17_56.png')
plt.show()

#分类报告
from sklearn.metrics import classification_report
predictions = model.predict_classes(X_test)
print(classification_report(truelabel,predictions))