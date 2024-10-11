import os
from tqdm import tqdm
import numpy as np
#from mypath import Path##这里是作者自己创建的一个文件，用来生成路径的
 
def calculate_weigths_labels(dataset, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']##这里是作者创建的一个dataloader，这里的sample['label']返回的是标签图像的lable mask
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)##统计每幅图像中不同类别像素的个数
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))##这里是计算每个类别像素的权重
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(os.path.dirname(dataset), dataset+'_classes_weights.npy')##生成权重文件
    np.save(classes_weights_path, ret)##把各类别像素权重保存到一个文件中
 
    return ret

#  训练数据标签路径
train_label_path = "E:/huan/tiffdatawulingyuan/splitdata/trainlabel/"
dataset1 = "E:/huan/tiffdatawulingyuan/splitdata/model/"
weight = calculate_weigths_labels(dataset1, train_label_path, 2)

#criterion=nn.CrossEntropyLoss(weight=self.weight,ignore_index=self.ignore_index, reduction='mean')