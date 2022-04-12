# HOR
Detecting Deepfake Videos using the Disharmony between Intra- and Inter-frame Maps
Detecting Deepfake Videos using the Disharmony between Intra- and Inter-frame Maps trai_CNN_TWOstream是主要训练文件。首先通过download-FaceForensics_v3下载数据集，然后通过FaceForensicsDataProcessing处理数据集，包括定位人脸以及裁剪。最后通过flow文件生成包含光流和 RGB的NPZ文件，然后参考generate_json来选取训练集和验证集进行训练。
