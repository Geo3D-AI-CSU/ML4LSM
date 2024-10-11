import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from seg_unet import unet
# from AttResUnet import Attention_ResUNet
#from Model.seg_hrnet import seg_hrnet
from dataProcess import trainGenerator, color_dict
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import datetime
import xlwt
import os
from skimage import io
import numpy as np
from sklearn.utils import class_weight
from seg_unet import focal
from sklearn.model_selection import GridSearchCV
import tensorflow as tf

'''
数据集相关参数
'''
#  训练数据图像路径
train_image_path = "E:/huan/tiffdatawulingyuan/splitdata2/traindataset/"
#  训练数据标签路径
train_label_path = "E:/huan/tiffdatawulingyuan/splitdata2/trainlabel/"
#  验证数据图像路径
validation_image_path = "E:/huan/tiffdatawulingyuan/splitdata2/testdataset/"
#  验证数据标签路径
validation_label_path = "E:/huan/tiffdatawulingyuan/splitdata2/testlabel/"



#  训练数据数目
# train_num = len(os.listdir(train_image_path))
# #  验证数据数目
# validation_num = len(os.listdir(validation_image_path))
# #  训练集每个epoch有多少个batch_size
# steps_per_epoch = train_num / batch_size
# #  验证集每个epoch有多少个batch_size
# validation_steps = validation_num / batch_size
#  标签的颜色字典,用于onehot编码
colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)


#  得到一个生成器，以batch_size的速率生成训练数据
train_Generator = trainGenerator(batch_size,
                                 train_image_path, 
                                 train_label_path,
                                 classNum ,
                                 colorDict_GRAY,
                                 input_size)

#  得到一个生成器，以batch_size的速率生成验证数据
validation_data = trainGenerator(batch_size,
                                 validation_image_path,
                                 validation_label_path,
                                 classNum,
                                 colorDict_GRAY,
                                 input_size)
#  定义模型
model = unet( input_size = input_size, 
             classNum = classNum, 
             learning_rate = learning_rate)
# #model = seg_hrnet(pretrained_weights = premodel_path, 
# #                  input_size = input_size, 
# #                  classNum = classNum, 
# #                  learning_rate = learning_rate)
#  打印模型结构
model.summary()
# #  回调函数
# #  val_loss连续10轮没有下降则停止训练
# early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
# #  当3个epoch过去而val_loss不下降时，学习率减半
# reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, verbose = 1)
# model_checkpoint = ModelCheckpoint(model_path,
#                                    monitor = 'loss',
#                                    verbose = 1,# 日志显示模式:0->安静模式,1->进度条,2->每轮一行
#                                    save_best_only = True)

# #  获取当前时间
# start_time = datetime.datetime.now()

# weights = {0: 0.50454072, 1: 55.55734012}

# #  模型训练
# history = model.fit_generator(train_Generator,
#                     steps_per_epoch = steps_per_epoch,
#                     epochs = epochs,
#                     callbacks = [early_stopping, model_checkpoint, model_checkpoint],
#                     validation_data = validation_data,
#                     validation_steps = validation_steps)

# Use scikit-learn to grid search the batch size and epochsimport numpyfrom sklearn.grid_search import GridSearchCVfrom keras.models import Sequentialfrom keras.layers import Densefrom keras.wrappers.scikit_learn import KerasClassifier# Function to create model, required for KerasClassifierdef create_model():
 # define function to display the results of the grid search
def display_cv_results(search_results):
    print('Best score = {:.4f} using {}'.format(search_results.best_score_, search_results.best_params_))
    means = search_results.cv_results_['mean_test_score']
    stds = search_results.cv_results_['std_test_score']
    params = search_results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('mean test accuracy +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stdev, param))

def load_dataset():
    # load dataset
    x_train = []
    y_train = []
    for x, y in train_Generator:
        x_train.append(x)
        y_train.append(y)
    x_test = []
    y_test = []
    for x, y in validation_data:
        x_test.append(x)
        y_test.append(y)
    (trainX, trainY), (testX, testY) = (x_train, y_train), (x_test, y_test)
    # # reshape dataset to have a single channel
    # trainX = trainX.reshape((trainX.shape[0], 128, 128, 3))
    # testX = testX.reshape((testX.shape[0], 128, 128, 3))
    # # one hot encode target values
    # trainY = tf.keras.utils.to_categorical(trainY)
    # testY = tf.keras.utils.to_categorical(testY)
    return trainX, trainY, testX, testY

x_train, y_train, x_test, y_test = load_dataset()
print(1)
# normalizing inputs from 0-255 to 0.0-1.0
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

num_classes = y_test.shape[1]

n_epochs = 50
n_epochs_cv = 10
n_cv = 3

param_grid = {
    'dropout_rate': [0.0, 0.10, 0.20, 0.30],
    'batch_size': [2, 8, 16],
    'epochs': [n_epochs_cv]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=n_cv)
grid_result = grid.fit(x_train, y_train)  # fit the full dataset as we are using cross validation

# print results
unet.display_cv_results(grid_result)



# # Use scikit-learn to grid search the batch size and epochsimport numpyfrom sklearn.grid_search import GridSearchCVfrom keras.models import Sequentialfrom keras.layers import Densefrom keras.wrappers.scikit_learn import KerasClassifier# Function to create model, required for KerasClassifierdef create_model(optimizer='adam'):
#     # create model
#     model = Sequential()
#     model.add(Dense(12, input_dim=8, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))    # Compile model
#     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])    return model    # fix random seed for reproducibilityseed = 7numpy.random.seed(seed)    # load datasetdataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")# split into input (X) and output (Y) variablesX = dataset[:,0:8]
# Y = dataset[:,8]# create modelmodel = KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=10, verbose=0)# define the grid search parametersoptimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# param_grid = dict(optimizer=optimizer)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
# grid_result = grid.fit(X, Y)# summarize resultsprint("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))for params, mean_score, scores in grid_result.grid_scores_:
#     print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

# # Use scikit-learn to grid search the learning rate and momentumimport numpyfrom sklearn.grid_search import GridSearchCVfrom keras.models import Sequentialfrom keras.layers import Densefrom keras.wrappers.scikit_learn import KerasClassifierfrom keras.optimizers import SGD# Function to create model, required for KerasClassifierdef create_model(learn_rate=0.01, momentum=0):
#     # create model
#     model = Sequential()
#     model.add(Dense(12, input_dim=8, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))    # Compile model
#     optimizer = SGD(lr=learn_rate, momentum=momentum)
#     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])    return model    # fix random seed for reproducibilityseed = 7numpy.random.seed(seed)    # load datasetdataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")    # split into input (X) and output (Y) variablesX = dataset[:,0:8]
# Y = dataset[:,8]    # create modelmodel = KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=10, verbose=0)    # define the grid search parameterslearn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
# momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
# param_grid = dict(learn_rate=learn_rate, momentum=momentum)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
# grid_result = grid.fit(X, Y)    # summarize resultsprint("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))for params, mean_score, scores in grid_result.grid_scores_:
#     print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))