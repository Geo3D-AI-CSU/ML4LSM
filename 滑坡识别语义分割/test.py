import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from seg_unet import unet

from dataProcess import testGenerator, saveResult, color_dict


#  训练模型保存地址
model_path = r"E:/huan/tiffdatawulingyuan/model/unet_model_epoch20_1e-4_focaldataset3.h5"
#  测试数据路径
test_iamge_path = r"E:/huan/tiffdatawulingyuan/splitdata2/testdataset/"
#  结果保存路径
save_path = r"E:/huan/tiffdatawulingyuan/predictdata/"
#  测试数据数目
test_num = len(os.listdir(test_iamge_path))
#  类的数目(包括背景)
classNum = 2
#  模型输入图像大小
input_size = (128, 128, 3)
#  生成图像大小
output_size = (128, 128)
#  训练数据标签路径
train_label_path = "E:/huan/tiffdatawulingyuan/splitdata2/testlabel/"
#  标签的颜色字典
colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)

model = unet(model_path)

testGene = testGenerator(test_iamge_path, input_size)

#  预测值的Numpy数组
results = model.predict_generator(testGene,
                                  test_num,
                                  verbose = 1)

#  保存结果
saveResult(test_iamge_path, save_path, results, colorDict_GRAY, output_size)