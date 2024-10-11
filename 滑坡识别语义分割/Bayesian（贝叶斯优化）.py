import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#from AttResUnet import Attention_ResUNet
#from Model.seg_hrnet import seg_hrnet
from dataProcess import trainGenerator, color_dict
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import datetime
import xlwt
import os
import tensorflow as tf
from skimage import io
import numpy as np
from sklearn.utils import class_weight
from seg_unet import focal
from sklearn.model_selection import GridSearchCV
# from keras.utils.np_utils import *
from tensorflow.keras import utils
import numpy as np 
import os
import random
import gdal
import cv2
from skimage import io
import keras_tuner as kt
from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.models import Model
# # from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
# from keras.layers import merge

from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.layers import concatenate

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
#from hyperopt import hp
from keras_tuner import HyperParameters, Hyperband, RandomSearch


# save_train_image = "E:/huan/tiffdatawulingyuan/splitdata2/traindataset/"
# save_train_label = "E:/huan/tiffdatawulingyuan/splitdata2/trainlabel/"
# save_validation_image = "E:/huan/tiffdatawulingyuan/splitdata2/testdataset/"
# save_validation_label = "E:/huan/tiffdatawulingyuan/splitdata2/testlabel/"

save_train_image = r"F:/Tangjiacheng-CSU/CNN4LSM/data/滑坡识别语义分割/tiffdatawulingyuan/splitdata2/traindataset/"
save_train_label = r"F:/Tangjiacheng-CSU/CNN4LSM/data/滑坡识别语义分割/tiffdatawulingyuan/splitdata2/trainlabel/"
save_validation_image = r"F:/Tangjiacheng-CSU/CNN4LSM/data/滑坡识别语义分割/tiffdatawulingyuan/splitdata2/testdataset/"
save_validation_label = r"F:/Tangjiacheng-CSU/CNN4LSM/data/滑坡识别语义分割/tiffdatawulingyuan/splitdata2/testlabel/"

'''
模型相关参数
'''
#  批大小
batch_size = 4
#  类的数目(包括背景)
classNum = 2
#  模型输入图像大小
input_size = (128, 128, 3)
#  训练模型的迭代总轮数
epochs = 1
#  初始学习率
learning_rate = 0.0001
#  预训练模型地址
premodel_path = None



#  获取颜色字典
#  labelFolder 标签文件夹,之所以遍历文件夹是因为一张标签可能不包含所有类别颜色
#  classNum 类别总数(含背景)
def color_dict(labelFolder, classNum):
    colorDict = []
    #  获取文件夹内的文件名
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + ImageNameList[i]
        img = io.imread(ImagePath).astype(np.uint32)
        #  如果是灰度，转成RGB
        if(len(img.shape) == 2):
            img = img = 255 * np.array(img).astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        #  为了提取唯一值，将RGB转成一个数
        img_new = img[:,:,0] * 1000000 + img[:,:,1] * 1000 + img[:,:,2]
        unique = np.unique(img_new)
        #  将第i个像素矩阵的唯一值添加到colorDict中
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  对目前i个像素矩阵里的唯一值再取唯一值
        colorDict = sorted(set(colorDict))
        #  若唯一值数目等于总类数(包括背景)ClassNum，停止遍历剩余的图像
        if(len(colorDict) == classNum):
            break
    #  存储颜色的RGB字典，用于预测时的渲染结果
    colorDict_RGB = []
    for k in range(len(colorDict)):
        #  对没有达到九位数字的结果进行左边补零(eg:5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  前3位R,中3位G,后3位B
        color_RGB = [int(color[0 : 3]), int(color[3 : 6]), int(color[6 : 9])]
        colorDict_RGB.append(color_RGB)
    #  转为numpy格式
    colorDict_RGB = np.array(colorDict_RGB)
    #  存储颜色的GRAY字典，用于预处理时的onehot编码
    colorDict_GRAY = colorDict_RGB.reshape((colorDict_RGB.shape[0], 1 ,colorDict_RGB.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_RGB, colorDict_GRAY

#  读取图像像素矩阵
#  fileName 图像文件名
def readTif(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
    return GdalImg_data

#  数据预处理：图像归一化+标签onehot编码
#  img 图像数据
#  label 标签数据
#  classNum 类别总数(含背景)
#  colorDict_GRAY 颜色字典
def dataPreprocess(img, label, classNum, colorDict_GRAY):
    #  归一化
    imageList = os.listdir(img)
    labelList = os.listdir(label)
    img = readTif(img + imageList[0])
    #  GDAL读数据是(BandNum,Width,Height)要转换为->(Width,Height,BandNum)
    img = img.swapaxes(1, 0)
    img = img.swapaxes(1, 2)
    label = readTif(label + labelList[0]).astype(np.uint8)
    label = label.swapaxes(1, 0)
    label = label.swapaxes(1, 2)
    img = img / 255.0
    label =label / 255.0
    # for i in range(colorDict_GRAY.shape[0]):
    #     label[label == colorDict_GRAY[i][0]] = i
    #  将数据厚度扩展到classNum层
    new_label = np.zeros(label.shape + (classNum,))
    #  将平面的label的每类，都单独变成一层
    for i in range(classNum):
        new_label[label == i,i] = 1                                          
    label = new_label
    return (img, label)

#  训练数据数目
train_num = len(os.listdir(save_train_image))
#  验证数据数目
validation_num = len(os.listdir(save_validation_image))
#  训练集每个epoch有多少个batch_size
steps_per_epoch = train_num / batch_size
#  验证集每个epoch有多少个batch_size
validation_steps = validation_num / batch_size
#  标签的颜色字典,用于onehot编码
colorDict_RGB, colorDict_GRAY = color_dict(save_train_label, classNum)

x_train,y_train = dataPreprocess(save_train_image, save_train_label, classNum, colorDict_GRAY)

x_test,y_test = dataPreprocess(save_validation_image, save_validation_label, classNum, colorDict_GRAY)

#  得到一个生成器，以batch_size的速率生成训练数据
train_Generator = trainGenerator(batch_size,
                                 save_train_image, 
                                 save_train_label,
                                 classNum ,
                                 colorDict_GRAY,
                                 input_size)

#  得到一个生成器，以batch_size的速率生成验证数据
validation_data = trainGenerator(batch_size,
                                 save_validation_image,
                                 save_validation_label,
                                 classNum,
                                 colorDict_GRAY,
                                 input_size)

# tr_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(1)
# te_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(1)
# x_train = cv2.resize(np.array(x_train), (128, 128))
# y_train = cv2.resize(np.array(y_train), (128, 128))
# x_test = cv2.resize(np.array(x_test), (128, 128))
# y_test = cv2.resize(np.array(y_test), (128, 128))
# x_train = np.expand_dims(x_train,axis=0)
# y_train = np.expand_dims(y_train,axis=0)
# x_test = np.expand_dims(x_test,axis=0)
# y_test = np.expand_dims(y_test,axis=0)
# x_train = save_train_image
# y_train = save_train_label
# x_test = save_validation_image
# y_test = save_validation_label

# # normalize pixels to range 0-1
# train_x = x_train / 255.0
# test_x = x_test / 255.0

# #one-hot encode target variable
# train_y = to_categorical(y_train)
# test_y = to_categorical(y_test)

# print(x_train.shape) #(57000, 28, 28)
# print(y_train.shape) #(57000, 10)
# print(x_test.shape) #(10000, 28, 28)
# print(y_test.shape) #(10000, 10)

print(1)

def build_model(hp):
    model = Sequential()
    inputs = Input(input_size)
    #  2D卷积层
    conv1 = BatchNormalization()(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs))
    conv1 = BatchNormalization()(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1))
    #  对于空间数据的最大池化
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = BatchNormalization()(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1))
    conv2 = BatchNormalization()(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = BatchNormalization()(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2))
    conv3 = BatchNormalization()(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = BatchNormalization()(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3))
    conv4 = BatchNormalization()(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4))
    #  Dropout正规化，防止过拟合
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
 
    conv5 = BatchNormalization()(Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4))
    conv5 = BatchNormalization()(Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5))
    drop5 = Dropout(0.5)(conv5)
    #  上采样之后再进行卷积，相当于转置卷积操作
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    
    try:
        merge6 = concatenate([drop4,up6],axis = 3)
    except:
        merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    conv6 = BatchNormalization()(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6))
    conv6 = BatchNormalization()(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6))
 
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    try:
        merge7 = concatenate([conv3,up7],axis = 3)
    except:
        merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = BatchNormalization()(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7))
    conv7 = BatchNormalization()(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7))
 
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    try:
        merge8 = concatenate([conv2,up8],axis = 3)
    except:
        merge8 = merge([conv2,up8],mode = 'concat', concat_axis = 3)
    conv8 = BatchNormalization()(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8))
    conv8 = BatchNormalization()(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8))
 
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    try:
        merge9 = concatenate([conv1,up9],axis = 3)
    except:
        merge9 = merge([conv1,up9],mode = 'concat', concat_axis = 3)
    conv9 = BatchNormalization()(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9))
    conv9 = BatchNormalization()(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9))
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(classNum, 1, activation = 'sigmoid')(conv9)
 
    model = Model(inputs = inputs, outputs = conv10)
 
    #  用于配置训练模型（优化器、目标函数、模型评估标准）

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    hp_optimizer=hp.Choice('Optimizer', values=['Adam', 'SGD'])

    if hp_optimizer == 'Adam':
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    elif hp_optimizer == 'SGD':
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        nesterov=True
        momentum=0.9
    weights = {0: 0.50454072, 1: 55.55734012}
    #hp = HyperParameters()
    model.compile(optimizer=hp_optimizer,
                  loss = focal(alpha = hp.Float('alpha', min_value=0,max_value=1), gamma = hp.Float('gamma', min_value=0,max_value=5)),
                  metrics=['accuracy'])
    # model.compile(optimizer= Adam(lr = learning_rate), loss = focal(), metrics=['accuracy'])
    return model
hp = HyperParameters()
#Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4]))
#hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4])
#loss = focal(alpha = hp.Float('alpha', min_value=0,max_value=1), gamma = hp.Float('gamma', min_value=0,max_value=5))
# model.compile(optimizer = hp_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#  回调函数
#  val_loss连续10轮没有下降则停止训练
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
#tuner_mlp = kt.tuners.BayesianOptimization(model, objective='val_loss', max_trials=30, directory='.', project_name='tuning-mlp')
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
history = model.fit(x_train, y_train, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

#最佳循环数不再进行训练
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hyper = model.fit(x_train, y_train, epochs=best_epoch, validation_split=0.2)

#测试数据上评估超模型
eval_result = hypermodel.evaluate(x_test, y_test)
print("[test loss, test accuracy]:", eval_result)