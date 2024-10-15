# 多尺度卷积神经网络(CNN)，用于遥感图像和滑坡敏感性制图 (LSM) 的滑坡清单制图

准确的滑坡易发性制图 （LSM） 依赖于详细的滑坡清单和相关影响因素。本研究采用 Sentinel 2 遥感影像，利用关注 U-Net 骨干网络在中国湖南省张家界市建立了全面的滑坡清单。随后，具有更精确边界的细化滑坡清单被整合到 LSM 流程中。在 LSM 中引入多尺度采样三维卷积神经网络 （3D-CNN），有助于提取相关地形、水文、气象、地质和人类活动因素的多尺度邻域特征和深度信息。

论文链接：[Multi-scale convolutional neural networks (CNNs) for landslide inventory mapping from remote sensing imagery and landslide susceptibility mapping](https://www.tandfonline.com/doi/full/10.1080/19475705.2024.2383309 "论文")

***

# 目录  
- [文件介绍](#文件介绍)  
- [环境配置](#环境配置)
- [数据](#数据)  
- [鸣谢](#鸣谢)

# 文件介绍
    该代码仓库基于共分为三个目录
    1.滑坡语义识别分割
    2.集成机器学习滑坡易发性评价
    3.卷积神经网络滑坡易发性评价

# 环境配置
    本环境配置旨在能够运行U-net和VGG网络实验
    由于第三方库存在依赖关系，故给出必须安装的内容
    GDAL==2.2.4
    tensorflow-gpu==2.6.0
    keras==2.6.0
    setuptools==57.5.0
    xgboost==1.3.3
    scikit-learn==0.23.2
    xlwt==1.3.0
    scikit-image==0.15.0
    opencv-python==3.4.10.37
    keras-tuner==1.4.6
    numpy
    pandas
    matplotlib
    
# 数据
由于与湖南省地质灾害调查监测研究院签订了数据隐私协议，本研究中使用的数据集未公开提供。但是，如果提出合理要求，可以从通讯作者处获得

# 鸣谢
本研究得到了湖南省自然科学基金（批准号 2022JJ30708 和 2023JJ60188）、湖南省自然资源科学和技术规划计划（批准号 20230123XX）、长沙市自然科学基金（批准号 KQ2208054）和国家自然科学基金（批准号 42072326）的资助
