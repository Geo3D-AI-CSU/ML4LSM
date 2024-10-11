import gdal
import numpy as np
from tensorflow.keras.models import load_model
# from keras import losses
import datetime
import math
import sys
from seg_unet import unet

#  读取tif数据集
def readTif(fileName, xoff = 0, yoff = 0, data_width = 0, data_height = 0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize 
    #  栅格矩阵的行数
    height = dataset.RasterYSize 
    #  波段数
    bands = dataset.RasterCount 
    #  获取数据
    if(data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj

#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset

#  tif裁剪（tif像素数据，裁剪边长）
def TifCroppingArray(img, SideLength):
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目
    ColumnNum = int((img.shape[0] - SideLength * 2) / (128 - SideLength * 2))
    #  行上图像块数目
    RowNum = int((img.shape[1] - SideLength * 2) / (128 - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (128 - SideLength * 2) : i * (128 - SideLength * 2) + 128,
                          j * (128 - SideLength * 2) : j * (128 - SideLength * 2) + 128]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[i * (128 - SideLength * 2) : i * (128 - SideLength * 2) + 128,
                      (img.shape[1] - 128) : img.shape[1]]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[(img.shape[0] - 128) : img.shape[0],
                      j * (128-SideLength*2) : j * (128 - SideLength * 2) + 128]
        TifArray.append(cropped)
    #  向前裁剪右下角
    cropped = img[(img.shape[0] - 128) : img.shape[0],
                  (img.shape[1] - 128) : img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[0] - SideLength * 2) % (128 - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[1] - SideLength * 2) % (128 - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver

#  标签可视化，即为第n类赋上n值
def labelVisualize(img):
    img_out = np.zeros((img.shape[0],img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #  为第n类赋上n值
            img_out[i][j] = np.argmax(img[i][j])
    return img_out

#  对测试图片进行归一化，并使其维度上和训练图片保持一致
def testGenerator(TifArray):
    for i in range(len(TifArray)):
        for j in range(len(TifArray[0])):
            img = TifArray[i][j]
            #  归一化
            img = img / 255.0
            #  在不改变数据内容情况下，改变shape
            img = np.reshape(img,(1,)+img.shape)
            yield img

#  获得结果矩阵
def Result(shape, TifArray, npyfile, num_class, RepetitiveLength, RowOver, ColumnOver):
    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0  
    for i,item in enumerate(npyfile):
        img = labelVisualize(item)
        img = img.astype(np.uint8)
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if(i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 128 - RepetitiveLength, 0 : 128-RepetitiveLength] = img[0 : 128 - RepetitiveLength, 0 : 128 - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                #  原来错误的
                #result[shape[0] - ColumnOver : shape[0], 0 : 256 - RepetitiveLength] = img[0 : ColumnOver, 0 : 256 - RepetitiveLength]
                #  后来修改的
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0 : 128 - RepetitiveLength] = img[128 - ColumnOver - RepetitiveLength : 128, 0 : 128 - RepetitiveLength]
            else:
                result[j * (128 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (128 - 2 * RepetitiveLength) + RepetitiveLength,
                       0:128-RepetitiveLength] = img[RepetitiveLength : 128 - RepetitiveLength, 0 : 128 - RepetitiveLength]   
        #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif(i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 128 - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0 : 128 - RepetitiveLength, 128 -  RowOver: 128]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0], shape[1] - RowOver : shape[1]] = img[128 - ColumnOver : 128, 128 - RowOver : 128]
            else:
                result[j * (128 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (128 - 2 * RepetitiveLength) + RepetitiveLength,
                       shape[1] - RowOver : shape[1]] = img[RepetitiveLength : 128 - RepetitiveLength, 128 - RowOver : 128]   
            #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 128 - RepetitiveLength,
                       (i - j * len(TifArray[0])) * (128 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (128 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[0 : 128 - RepetitiveLength, RepetitiveLength : 128 - RepetitiveLength]         
            #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0],
                       (i - j * len(TifArray[0])) * (128 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (128 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[128 - ColumnOver : 128, RepetitiveLength : 128 - RepetitiveLength]
            else:
                result[j * (128 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (128 - 2 * RepetitiveLength) + RepetitiveLength,
                       (i - j * len(TifArray[0])) * (128 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (128 - 2 * RepetitiveLength) + RepetitiveLength,
                       ] = img[RepetitiveLength : 128 - RepetitiveLength, RepetitiveLength : 128 - RepetitiveLength]
    return result

area_perc = 0.5
TifPath = r"E:/huan/tiffdatawulingyuan/originaldata/zzjbigarea.tif"
ModelPath = r"E:/huan/tiffdatawulingyuan/model/unet_modelwly_epoch20_1e-4_weight.h5"
ResultPath = r"E:/huan/tiffdatawulingyuan/bigareapre_result12.tif"
#TifPath = sys.argv[1]
#ModelPath = sys.argv[2]
#ResultPath = sys.argv[3]
#area_perc = float(sys.argv[4])
RepetitiveLength = int((1 - math.sqrt(area_perc)) * 128 / 2)

#  记录测试消耗时间
testtime = []
#  获取当前时间
starttime = datetime.datetime.now()

im_width, im_height, im_bands, im_data, im_geotrans, im_proj = readTif(TifPath)
im_data = im_data.swapaxes(1, 0)
im_data = im_data.swapaxes(1, 2)

TifArray, RowOver, ColumnOver = TifCroppingArray(im_data, RepetitiveLength)
endtime = datetime.datetime.now()
text = "读取tif并裁剪预处理完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"
print(text)
testtime.append(text)

model = unet(ModelPath)
testGene = testGenerator(TifArray)
results = model.predict_generator(testGene,
                                  len   (TifArray) * len(TifArray[0]),
                                  verbose = 1)
endtime = datetime.datetime.now()
text = "模型预测完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"
print(text)
testtime.append(text)

#保存结果
result_shape = (im_data.shape[0], im_data.shape[1])
result_data = Result(result_shape, TifArray, results, 2, RepetitiveLength, RowOver, ColumnOver)
writeTiff(result_data, im_geotrans, im_proj, ResultPath)
endtime = datetime.datetime.now()
text = "结果拼接完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"
print(text)
testtime.append(text)

time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
with open('timelog_%s.txt'%time, 'w') as f:
    for i in range(len(testtime)):
        f.write(testtime[i])
        f.write("\r\n")