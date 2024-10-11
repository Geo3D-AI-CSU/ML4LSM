from PIL import Image
from osgeo import gdal
import os
import json

# 以tiff转png为例，其他格式同理，
# 代码中路径更改为自己图像存放路径即可

imagesDirectory= r"E:/huan/tiffdata/splitdata/test/label"  # tiff图片所在文件夹路径
distDirectory = os.path.dirname(imagesDirectory)
distDirectory = os.path.join(distDirectory, "E:/huan/tiffdata/splitdata/test/labelpng")# 要存放png格式的文件夹路径
for imageName in os.listdir(imagesDirectory):
    imagePath = os.path.join(imagesDirectory, imageName)
    image = Image.open(imagePath)# 打开tiff图像
    distImagePath = os.path.join(distDirectory, imageName[:-4]+'.png')# 更改图像后缀为.png，与原图像同名
    if image.mode == "F":
        image = image.convert('RGB')
    image.save(distImagePath)# 保存png图像

imagesDirectory1= r"E:/huan/tiffdatawulingyuan/predictdata"  # tiff图片所在文件夹路径
distDirectory1 = os.path.dirname(imagesDirectory1)
distDirectory1 = os.path.join(distDirectory1, "E:/huan/tiffdata/splitdata/test/predict1png")# 要存放png格式的文件夹路径
for imageName in os.listdir(imagesDirectory1):
    imagePath = os.path.join(imagesDirectory1, imageName)
    image = Image.open(imagePath)# 打开tiff图像
    distImagePath = os.path.join(distDirectory1, imageName[:-4]+'.png')# 更改图像后缀为.png，与原图像同名
    if image.mode == "F":
        image = image.convert('RGB')
    image.save(distImagePath)# 保存png图像

# 获取混淆矩阵四个值（以道路提取为例，道路区域【255,255,255】，背景区域【0,0，0】）
# TP：被模型预测为正类的正样本(预测道路且标签道路)
# TN：被模型预测为负类的负样本（预测背景且真实背景）
# FP：被模型预测为正类的负样本（预测道路但真实背景）
# FN：被模型预测为负类的正样本（预测背景但真实道路）
def get_vaslue(predict_folders_path, label_folders_path):
    # 加载文件夹
    #################################################################
    #  标签图像文件夹
    label_folders_path = r"E:/huan/tiffdata/splitdata/test/labelpng/"
    #  预测图像文件夹
    predict_folders_path = r"E:/huan/tiffdata/splitdata/test/predict1png/"
    #################################################################
    predict_folders = os.listdir(predict_folders_path)
    label_folders = os.listdir(label_folders_path)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for folder in predict_folders:
        # 获取图片路径
        predict_folder_path = os.path.join(predict_folders_path, folder)
        label_folder_path = os.path.join(label_folders_path, folder)
        # 加载图像并赋值四通道
        predict = Image.open(predict_folder_path)
        predict = predict.convert('RGBA')
        label = Image.open(label_folder_path)
        label = label.convert('RGBA')
        heigh, width = predict.size
        # save_name = str(folder).split('.')[0]
        for i in range(heigh):
            for j in range(width):
                r_1, g_1, b_1, a_1 = predict.getpixel((i, j))
                r_2, g_2, b_2, a_2 = label.getpixel((i, j))
                if r_1 == 255:
                    if r_2 == 255:
                        TP += 1
                    if r_2 == 0:
                        FP += 1
                if r_1 == 0:
                    if r_2 == 255:
                        FN += 1
                    if r_2 == 0:
                        TN += 1
    return float(TP), float(TN), float(FP), float(FN)


# list转存txt
def list2txt(list, save_path, txt_name):
    with open(save_path + r'/' + txt_name, 'w') as f:
        json.dump(list, f)


def evoluation(TP, TN, FP, FN):
    evo = []
    # 准确率
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # 精准率
    #precision = TP / (TP + FP)
    # 召回率
    recall = TP / (TP + FN)
    # miou
    miou = (TP / (TP + FP + FN) + TN / (TN + FN + FP)) / 2
    # F1
    #f1 = 2 * ((precision * recall) / (precision + recall))
    evo.append('accuracy:{}  recall:{}  miou:{}'.format(accuracy, recall, miou))
    print(evo)
    list2txt(evo, r"", '')
    return evo


if __name__ == '__main__':
    predict_path = r''
    label_path = r''
    TP, TN, FP, FN = get_vaslue(predict_path, label_path)
    evoluation(TP, TN, FP, FN)
