import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore") #过滤掉警告的意思
from pyforest import *
data=pd.read_csv(r"E:/huan/ml/data/new/new01_1.csv",encoding='gb18030')
data.head()
print(data)
#图片显示中文
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus'] =False #减号unicode编码
data.drop(['OBJECTID','x','y','remote1'], axis=1, inplace=True) #删除无关的列
#data.columns = ['距河流的距离','距道路的距离','NDVI','坡向','平面曲率','剖面曲率','TWI','土地利用/土地覆盖','海拔','地形起伏度','地形粗糙度','降雨量','坡度','岩性','土壤性质']
#计算各变量之间的相关系数
corr = data.corr()
corr
print(corr)
ax = plt.subplots(figsize=(10, 10))#调整画布大小
ax = sns.heatmap(corr, vmax=.9, square=True, annot=True, cmap="YlGnBu")#画热力图   annot=True 表示显示系数
# 设置刻度字体大小
plt.xticks(fontsize=20,rotation=45)
plt.yticks(fontsize=20,rotation=360)
plt.show()
