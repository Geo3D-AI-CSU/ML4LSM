#!/usr/bin/env python
# coding: utf-8

# In[1]:
from matplotlib import pyplot as plt
# 封装好的KMeans聚类包
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import pandas as pd
# 1. 数据准备
# 假设已经准备好数据集，特征保存在X矩阵中，目标变量保存在y向量中
# Load dataset
data = pd.read_csv(r'E:/huan/ml/data/new/new01_1.csv')
data = data.fillna(0)
X = data.iloc[:,np.r_[1:13,16:17]].values
org_x=X
minmax = preprocessing.MinMaxScaler()
X = minmax.fit_transform(X)
y = data['remote1'].values 
print(X,y)

# 2. 数据标准化
# 对特征进行标准化，确保不同特征具有相同的尺度，以避免某些特征对聚类结果的影响过大

# 3. 聚类算法
# 使用K均值聚类算法将数据集划分为具有相似特征或目标的局部区域

# 设置聚类的区域数
num_clusters = 3

# 创建K均值聚类对象
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# 对数据集进行聚类
cluster_labels = kmeans.fit_predict(X)

# 4. 获取每个样本所属的局部区域
# 获取每个样本所属的聚类标签
sample_labels = kmeans.labels_
sample_centers=kmeans.cluster_centers_

# 5. 划分局部区域的训练数据和测试数据
train_X = []
train_y = []
test_X = []
test_y = []

for cluster_label in range(num_clusters):
    # 获取属于当前局部区域的样本索引
    samples_in_cluster = np.where(sample_labels == cluster_label)[0]
    np.random.shuffle(samples_in_cluster)
    # 划分训练集和测试集
    train_indices = samples_in_cluster[:int(0.8 * len(samples_in_cluster))]  # 前80%作为训练集
    test_indices = samples_in_cluster[int(0.8 * len(samples_in_cluster)):]  # 后20%作为测试集
    
    # 根据索引获取训练数据和测试数据
    train_X.append(X[train_indices])
    train_y.append(y[train_indices])
    test_X.append(X[test_indices])
    test_y.append(y[test_indices])

local_train_X=np.array(train_X)
local_train_y=np.array(train_y)
local_test_X=np.array(test_X)
local_test_y=np.array(test_y)

# In[2]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# 1. 数据划分
# 假设已经将数据集按照局部区域划分，并得到每个局部区域的训练数据和测试数据

# 2. 模型训练
# 针对每个局部区域，分别训练随机森林、GBDT和XGBoost模型

# 随机森林模型
rf_model = RandomForestClassifier(max_depth=20, max_features=0.8, max_leaf_nodes=45, n_estimators=1100)
rf_model.fit(local_train_X[0], local_train_y[0])

# GBDT模型
gbdt_model = GradientBoostingClassifier(max_depth= 34, max_features=0.9, max_leaf_nodes=375, n_estimators=43)
gbdt_model.fit(local_train_X[1], local_train_y[1])

# XGBoost模型
xgb_model = xgb.XGBClassifier(max_depth=9, n_estimators=145, reg_alpha=0.9, reg_lambda=0.8,eval_metric=['logloss','auc','error'],learning_rate=0.1,n_jobs=-1)
xgb_model.fit(local_train_X[2], local_train_y[2])

##
# 3. 模型预测
# 对于新的输入样本，首先确定其所属的局部区域，然后使用相应的局部模型进行预测
#%%

# rf_predictions = rf_model.predict(local_test_X[0])
# gbdt_predictions = gbdt_model.predict(local_test_X[1])
# xgb_predictions = xgb_model.predict(local_test_X[2])

#%%
# 4. 结果融合
# 可以使用简单的平均或加权平均策略将局部预测结果融合，得到最终的整体预测结果
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score

# from sklearn.metrics import mean_squared_error #MSE
# from sklearn.metrics import mean_absolute_error #MAE
# from sklearn.metrics import r2_score#R 2
# from sklearn.metrics import cohen_kappa_score#Kappa

# ensemble_predictions=np.append(rf_predictions,gbdt_predictions)
# ensemble_predictions=np.append(ensemble_predictions,xgb_predictions)
# # 最终的整体预测结果
# ensemble_trueValue=np.append(local_test_y[0],local_test_y[1])
# ensemble_trueValue=np.append(ensemble_trueValue,local_test_y[2])

# accuracy=accuracy_score(ensemble_trueValue,ensemble_predictions)
# print("Model RandomForestClassifier,Accuracy %0.6f:"%(accuracy_score(ensemble_trueValue,ensemble_predictions)))
# precision = precision_score(ensemble_trueValue,ensemble_predictions)
# recall = recall_score(ensemble_trueValue,ensemble_predictions)
# f1score = f1_score(ensemble_trueValue,ensemble_predictions)
# print("precision=%f recall=%f f1score=%f"%(precision, recall, f1score))
# MSE=mean_squared_error(ensemble_trueValue,ensemble_predictions)
# MAE=mean_absolute_error(ensemble_trueValue,ensemble_predictions)
# RMSE=np.sqrt(mean_squared_error(ensemble_trueValue,ensemble_predictions))
# R2=r2_score(ensemble_trueValue,ensemble_predictions)
# Kappa=cohen_kappa_score(ensemble_trueValue, ensemble_predictions)
# print("MSE=%f MAE=%f RMSE=%f R2=%f Kappa=%f"%(MSE,MAE,RMSE,R2,Kappa))
# #report (clf2,local_test_X,local_test_y)

# from sklearn import metrics
# import pylab as plt

# Font={'size':18, 'family':'Times New Roman'}


# y_probas1 = rf_model.predict_proba(local_test_X[0])
# y_probas2 = gbdt_model.predict_proba(local_test_X[1])
# y_probas3 = xgb_model.predict_proba(local_test_X[2])

# y_scores1 = y_probas1[:,1]
# y_scores2 = y_probas2[:,1]
# y_scores3 = y_probas3[:,1]
# y_scores=np.append(y_scores1 ,y_scores2)
# y_scores=np.append(y_scores ,y_scores3)

# fpr,tpr,thres = metrics.roc_curve(ensemble_trueValue, y_scores,drop_intermediate=False)


# roc_auc = metrics.auc(fpr, tpr)


# print(roc_auc)
  
# plt.figure(figsize=(6,6))
# plt.plot(fpr, tpr, 'b', label = 'LL-stacking = %0.4f' % roc_auc, color='blue')

# plt.legend(loc = 'lower right', prop=Font)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('真阳性率', Font)
# plt.xlabel('假阳性率', Font)
# plt.tick_params(labelsize=15)
# plt.show()

# %%
data = pd.read_csv(r'E:/huan/ml/data/new/zjjdata1.csv')
data = data.fillna(0)
X_pre= data.iloc[:,np.r_[1:13,16:17]].values


y_label_new1_predict1 = rf_model.predict_proba(X_pre)[:,1]
y_label_new1_predict2 = gbdt_model.predict_proba(X_pre)[:,1]
y_label_new1_predict3 = xgb_model.predict_proba(X_pre)[:,1]
pd.DataFrame(y_label_new1_predict1).to_csv(r'E:/huan/ml/data/new/rf_model.csv',index=False)
pd.DataFrame(y_label_new1_predict1).to_csv(r'E:/huan/ml/data/new/gbdt_model.csv',index=False)
pd.DataFrame(y_label_new1_predict1).to_csv(r'E:/huan/ml/data/new/xgb_model.csv',index=False)

# pd.DateOffset(y_label_new1_predict1).to_csv(r'E:/huan/ml/data/new/rf_model.csv',index=False)
# pd.DateOffset(y_label_new1_predict1).to_csv(r'E:/huan/ml/data/new/gbdt_model.csv',index=False)
# pd.DateOffset(y_label_new1_predict1).to_csv(r'E:/huan/ml/data/new/xgb_model.csv',index=False)
#%%
#(rf_predictions[1,:] + gbdt_predictions[1,:] + xgb_predictions[1,:]) / 3
y_predict=(y_label_new1_predict1+y_label_new1_predict2+y_label_new1_predict3)/3.0
#y_predict=y_predict1.mean(axis=1)

# y_predict1=np.append(y_label_new1_predict1[:,1],y_label_new1_predict2[:,1])
# y_predict1=np.append(y_predict1,y_label_new1_predict3[:,1])

# 

# for indexes in indicesAll:
#     for i,j in enumerate(indexes):
#         y_predict[j]=y_predict1[i]

print('The New point 1 predict class:\n',y_predict)
result = pd.DataFrame(y_predict)
data['ll6']=result
#result=pd.DataFrame(columns=['yuce'], data=y_predict1)
data.to_csv(r'E:/huan/ml/data/new/zjjdata2.csv',mode = 'a',index=False)
print(data)
# %%
