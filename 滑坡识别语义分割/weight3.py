import numpy as np
import cv2
import os
from skimage import io

""" 
混淆矩阵
P\L     P    N 
P      TP    FP 
N      FN    TN 
"""  
#  获取颜色字典
#  labelFolder 标签文件夹,之所以遍历文件夹是因为一张标签可能不包含所有类别颜色
#  classNum 类别总数(含背景)
def color_dict(labelFolder, classNum):
    colorDict = []
    #  获取文件夹内的文件名
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        #ImagePath = labelFolder + "/" + ImageNameList[i]
        ImagePath = labelFolder + ImageNameList[i]
        #img = cv2.imread(r'E:\\huan\\tiffdata\\splitdata\\test\\label\\283.tif')
        img = io.imread(ImagePath).astype(np.uint32)
        #img = cv2.imread(ImagePath).astype(np.uint32)
        #  如果是灰度，转成RGB
        # if(len(img.shape) == 2):
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        #  为了提取唯一值，将RGB转成一个数
        img_new = img[:,:] * 1000000 + img[:,:] * 1000 + img[:,:]
        unique = np.unique(img_new)
        #  将第i个像素矩阵的唯一值添加到colorDict中
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  对目前i个像素矩阵里的唯一值再取唯一值
        colorDict = sorted(set(colorDict))
        #  若唯一值数目等于总类数(包括背景)ClassNum，停止遍历剩余的图像
        if(len(colorDict) == classNum):
            break
    #  存储颜色的BGR字典，用于预测时的渲染结果
    colorDict_BGR = []
    for k in range(len(colorDict)):
        #  对没有达到九位数字的结果进行左边补零(eg:5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  前3位B,中3位G,后3位R
        color_BGR = [int(color[0 : 3]), int(color[3 : 6]), int(color[6 : 9])]
        colorDict_BGR.append(color_BGR)
    #  转为numpy格式
    colorDict_BGR = np.array(colorDict_BGR)
    #  存储颜色的GRAY字典，用于预处理时的onehot编码
    colorDict_GRAY = colorDict_BGR.reshape((colorDict_BGR.shape[0], 1 ,colorDict_BGR.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_BGR, colorDict_GRAY

# def get_weight(numClass, pixel_count):
#    pixel_count = np.zeros((numClass,1))
#    W = 1 / np.log(pixel_count)
#    W = numClass * W / np.sum(W)
#    return W


#################################################################
#  标签图像文件夹
LabelPath = r"E:/huan/tiffdatawulingyuan/splitdata/testlabel/"
#  类别数目(包括背景)
classNum = 2
#################################################################

#  获取类别颜色字典
colorDict_BGR, colorDict_GRAY = color_dict(LabelPath, classNum)

#  获取文件夹内所有图像
labelList = os.listdir(LabelPath)

#  读取第一个图像，后面要用到它的shape
Label0 = cv2.imread(LabelPath + "//" + labelList[0], 0)

#  图像数目
label_num = len(labelList)

#  把所有图像放在一个数组里
label_all = np.zeros((label_num, ) + Label0.shape, np.uint8)
for i in range(label_num):
    Label = cv2.imread(LabelPath + "//" + labelList[i])
    Label = cv2.cvtColor(Label, cv2.COLOR_BGR2GRAY)
    label_all[i] = Label

#  把颜色映射为0,1,2,3...
for i in range(colorDict_GRAY.shape[0]):
    label_all[label_all == colorDict_GRAY[i][0]] = i

#  拉直成一维
label_all = label_all.flatten()


from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',np.unique(label_all),label_all)
#class_weight = get_weight(numClass=2, pixel_count=label_all)
print(class_weights)