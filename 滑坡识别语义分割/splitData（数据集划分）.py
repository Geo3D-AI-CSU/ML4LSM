# *_*coding: utf-8 *_*
# Author --LiMing--

import os
import random
import shutil
import time

def copyFile(dataFileDir, labelFileDir):
    data_image_list = os.listdir(dataFileDir) # 获取图片的原始路径
    label_image_list = os.listdir(labelFileDir) # 获取图片的原始路径
    data_image_list.sort(key=lambda x: int(x.split('.')[0]))
    label_image_list.sort(key=lambda x: int(x.split('.')[0]))

    image_number = len(data_image_list)
    train_number = int(image_number * train_rate)
    train_sample = random.sample(data_image_list, train_number) # 从image_list中随机获取0.8比例的图像.
    test_sample = list(set(data_image_list) - set(train_sample))
    sample = [train_sample, test_sample]

    # 复制图像到目标文件夹
    for k in range(len(save_dir)):
        if os.path.isdir(save_dir[k] + 'dataset'):
            for name in sample[k]:
                shutil.copy(os.path.join(dataFileDir, name), os.path.join(save_dir[k] + 'dataset'+'/', name))
        else:
            os.makedirs(save_dir[k] + 'dataset')
            for name in sample[k]:
                shutil.copy(os.path.join(dataFileDir, name), os.path.join(save_dir[k] + 'dataset'+'/', name))
        if os.path.isdir(save_dir[k] + 'label'):
            for name in sample[k]:
                shutil.copy(os.path.join(labelFileDir, name), os.path.join(save_dir[k] + 'label'+'/', name))
        else:
            os.makedirs(save_dir[k] + 'label')
            for name in sample[k]:
                shutil.copy(os.path.join(labelFileDir, name), os.path.join(save_dir[k] + 'label'+'/', name))

if __name__ == '__main__':
    time_start = time.time()

    # 原始数据集路径
    origion_path = 'E:/huan/tiffdatawulingyuan/smalldata/fourth/'

    # 保存路径
    save_train_dir = 'E:/huan/tiffdatawulingyuan/splitdata4/train'
    save_test_dir = 'E:/huan/tiffdatawulingyuan/splitdata4/test'
    save_dir = [save_train_dir, save_test_dir]

    # 训练集比例
    train_rate = 0.8

    # 数据集类别及数量
    file_list = os.listdir(origion_path)
    num_classes = len(file_list)

    dataFileDir=os.path.join(origion_path, 'dataset')
    labelFileDir=os.path.join(origion_path, 'label')
    copyFile(dataFileDir, labelFileDir)
    print('划分完毕！')
    # for i in range(num_classes):
    #     class_name = file_list[i]
    #     image_Dir = os.path.join(origion_path, class_name)
    #     copyFile(image_Dir, class_name)
        

    time_end = time.time()
    print('---------------')
    print('训练集和测试集划分共耗时%s!' % (time_end - time_start))