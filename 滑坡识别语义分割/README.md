# 滑坡识别语义分割
## 代码目录结构
- [影像处理相关](#影像处理相关)  
- [数据预处理](#数据预处理)  
- [模型Model](#模型Model)  
- [精度评价](#精度评价)  
- [模型训练](#模型训练)  
- [结果导出](#结果导出)
### 影像处理相关
#### fenge（影像分割）.py  
    主要用于影像的裁剪  
#### zengqiang（影像增强）.py
    主要用于将tif影像进行几何变换实现数据增强
#### tiftopng（tif转png）.py
    tif文件转换为png文件
#### filerename（把一个文件夹中的影像名中a去除）.py 
#### fileresort（把一个文件夹中的影像按a+数字顺序排列）.py
### 数据预处理
#### splitData（数据集划分）.py
    数据集拆分工具，划分比例0.8
#### dataProcess.py
    训练依赖，用于将影像进行RGB和灰度编码
#### weight（为给二类按像素量加权）.py
#### weight2.py
#### weight3.py
    为权重失衡服务，用于平衡损失函数类别失衡
### 模型Model
#### AttResUnet.py
    backbone网络
#### seg_unet.py
    早期使用的u-net网络
#### dice_loss.py
    混合损失函数
#### focal_loss.py
    focal_loss损失函数
#### Bayesian（贝叶斯优化）.py
    优化目标为早期的u-net网络，非最终版本
#### girdsearch.py
    格网搜索，已弃用
### 精度评价
##### seg_metrics（精度评价）.py
    计算混淆矩阵、精确度、召回率、F1-score、整体精度、IoU、mIoU、FWIoU
#### metrics（精度评价也可用这个）.py
    同样进行精度评价，与seg_metrics相似
#### xiangguanxishu（相关系数）.py
    无用，这是为后续滑坡影响因子服务的
### 模型训练
#### train.py
    模型训练使用函数
#### test.py
    测试数据输出
### 结果导出
#### bigareapredict（大范围预测）.py
    tif文件结果预测

