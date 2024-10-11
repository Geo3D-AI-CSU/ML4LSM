import os
from PIL import Image


#  训练数据图像路径
train_image_path = "E:/huan/tiffdatawulingyuan/splitdata2/traindataset/"
#  训练数据标签路径
train_label_path = "E:/huan/tiffdatawulingyuan/splitdata2/trainlabel/"
#  验证数据图像路径
validation_image_path = "E:/huan/tiffdatawulingyuan/splitdata2/testdataset/"
#  验证数据标签路径
validation_label_path = "E:/huan/tiffdatawulingyuan/splitdata2/testlabel/"

save_train_image = "E:/huan/tiffdatawulingyuan/splitdata2png/traindataset/"
save_train_label = "E:/huan/tiffdatawulingyuan/splitdata2png/trainlabel/"
save_validation_image = "E:/huan/tiffdatawulingyuan/splitdata2png/testdataset/"
save_validation_label = "E:/huan/tiffdatawulingyuan/splitdata2png/testlabel/"

counts = 0
files = os.listdir(validation_label_path)
for file in files:
    if file.endswith('tif'):
        tif_file = os.path.join(validation_label_path, file)

        file = file[:-3] + 'png'
        png_file = os.path.join(save_validation_label, file)
        im = Image.open(tif_file)
        im.save(png_file)
        print(png_file)
        counts += 1

print('%d done' %counts)